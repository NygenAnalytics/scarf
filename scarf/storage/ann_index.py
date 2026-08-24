import hashlib
import os
import tempfile
from typing import Any

import numpy as np
import zarr

from .types import as_zarr_array
from .layout import _group_zarr_format, get_compressors
from .profiles import StorageProfile

ANN_INDEX_ARRAY = "ann_idx_bytes"
ANN_INDEX_CHUNK_BYTES = 8 * 1024 * 1024
ANN_INDEX_FORMAT_VERSION = 1
_ANN_INDEX_METADATA = (
    "ann_index_format_version",
    "metric",
    "dimensions",
    "element_count",
    "payload_sha256",
)


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
    *,
    profile: StorageProfile,
    metric: str,
    dimensions: int,
    element_count: int,
    name: str = ANN_INDEX_ARRAY,
) -> None:
    """Persist an hnswlib index as a chunked byte array."""
    if int(ann_idx.get_current_count()) != int(element_count):
        raise ValueError("ANN index element count does not match its coordinates")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        ann_idx.save_index(path)
        byte_length = os.path.getsize(path)
        if name in group:
            del group[name]
        chunk_size = min(ANN_INDEX_CHUNK_BYTES, max(byte_length, 1))
        zarr_format = _group_zarr_format(group)
        array = group.create_array(
            name,
            shape=(byte_length,),
            chunks=(chunk_size,),
            dtype="uint8",
            overwrite=True,
            compressors=get_compressors(
                profile,
                zarrFormat=zarr_format,
            ),
        )
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for start in range(0, byte_length, ANN_INDEX_CHUNK_BYTES):
                values = np.frombuffer(
                    source.read(min(ANN_INDEX_CHUNK_BYTES, byte_length - start)),
                    dtype=np.uint8,
                )
                digest.update(values)
                array[start : start + len(values)] = values
        array.attrs["byte_length"] = byte_length
        array.attrs["ann_index_format_version"] = ANN_INDEX_FORMAT_VERSION
        array.attrs["metric"] = str(metric)
        array.attrs["dimensions"] = int(dimensions)
        array.attrs["element_count"] = int(element_count)
        array.attrs["payload_sha256"] = digest.hexdigest()
    finally:
        os.unlink(path)


def load_ann_index(
    group: zarr.Group,
    space: str,
    dim: int,
    expected_count: int | None = None,
    name: str = ANN_INDEX_ARRAY,
) -> Any:
    """Load an hnswlib index from a Zarr byte array."""
    import hnswlib

    if name not in group:
        raise FileNotFoundError(f"ANN index array {name!r} not found in group")
    source = as_zarr_array(group[name], name=name)
    if source.ndim != 1 or np.dtype(source.dtype) != np.dtype(np.uint8):
        raise ValueError("ANN index payload must be a one-dimensional uint8 array")
    stored_byte_length = source.attrs.get("byte_length")
    if stored_byte_length is not None:
        if isinstance(stored_byte_length, bool) or not isinstance(
            stored_byte_length, int
        ):
            raise ValueError("ANN index byte length is invalid")
        if stored_byte_length != int(source.shape[0]):
            raise ValueError("ANN index byte length does not match its payload")
    present_metadata = {key for key in _ANN_INDEX_METADATA if key in source.attrs}
    if present_metadata and present_metadata != set(_ANN_INDEX_METADATA):
        raise ValueError("ANN index metadata is incomplete")
    stored_count: int | None = None
    stored_digest: str | None = None
    if present_metadata:
        stored_version = source.attrs["ann_index_format_version"]
        stored_metric = source.attrs["metric"]
        stored_dimensions = source.attrs["dimensions"]
        stored_element_count = source.attrs["element_count"]
        payload_digest = source.attrs["payload_sha256"]
        if (
            isinstance(stored_version, bool)
            or not isinstance(stored_version, int)
            or stored_version != ANN_INDEX_FORMAT_VERSION
        ):
            raise ValueError("ANN index format version is unsupported")
        if not isinstance(stored_metric, str) or stored_metric != space:
            raise ValueError("ANN index metric does not match artifact provenance")
        if (
            isinstance(stored_dimensions, bool)
            or not isinstance(stored_dimensions, int)
            or stored_dimensions != dim
        ):
            raise ValueError("ANN index dimensions do not match artifact provenance")
        if isinstance(stored_element_count, bool) or not isinstance(
            stored_element_count, int
        ):
            raise ValueError("ANN index element count is invalid")
        stored_count = stored_element_count
        if expected_count is not None and stored_count != int(expected_count):
            raise ValueError("ANN index element count does not match coordinates")
        if not isinstance(payload_digest, str):
            raise ValueError("ANN index payload digest is invalid")
        stored_digest = payload_digest
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        digest = hashlib.sha256()
        with open(path, "wb") as destination:
            for start in range(0, int(source.shape[0]), ANN_INDEX_CHUNK_BYTES):
                values = np.asarray(
                    source[
                        start : min(
                            start + ANN_INDEX_CHUNK_BYTES,
                            int(source.shape[0]),
                        )
                    ],
                    dtype=np.uint8,
                )
                digest.update(values)
                destination.write(memoryview(values))
        if stored_digest is not None and digest.hexdigest() != stored_digest:
            raise ValueError("ANN index payload digest does not match its metadata")
        index = hnswlib.Index(space=space, dim=dim)
        index.load_index(path)
        actual_count = int(index.get_current_count())
        required_count = stored_count if expected_count is None else int(expected_count)
        if required_count is not None and actual_count != required_count:
            raise ValueError("ANN index element count does not match coordinates")
        return index
    finally:
        os.unlink(path)


def load_ann_index_from_path(
    path: str,
    space: str,
    dim: int,
    expected_count: int | None = None,
) -> Any:
    """Load an hnswlib index from a legacy filesystem path."""
    import hnswlib

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(path)
    if expected_count is not None and int(index.get_current_count()) != int(
        expected_count
    ):
        raise ValueError("ANN index element count does not match coordinates")
    return index
