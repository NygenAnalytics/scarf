import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .types import as_zarr_array
from .layout import _group_zarr_format, get_compressors
from .profiles import StorageProfile

ANN_INDEX_ARRAY = "ann_idx_bytes"
ANN_INDEX_CHUNK_BYTES = 8 * 1024 * 1024
# Reads and writes move several chunks per call so Zarr transfers them
# concurrently; memory stays bounded by one window.
ANN_INDEX_IO_BYTES = 8 * ANN_INDEX_CHUNK_BYTES
ANN_INDEX_FORMAT_VERSION = 1
_ANN_INDEX_METADATA = (
    "byte_length",
    "ann_index_format_version",
    "metric",
    "dimensions",
    "element_count",
    "payload_sha256",
)


@dataclass(frozen=True, slots=True)
class _ValidatedAnnIndexPayload:
    source: zarr.Array
    stored_count: int
    stored_digest: str


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def has_ann_index(group: zarr.Group, name: str = ANN_INDEX_ARRAY) -> bool:
    """Return whether a group contains an ANN index byte array."""
    return name in group


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
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(ANN_INDEX_CHUNK_BYTES), b""):
                digest.update(block)
        # The local file is hashed first so the array and its record are
        # created with one metadata write; overwrite replaces an older index.
        array = group.create_array(
            name,
            shape=(byte_length,),
            chunks=(min(ANN_INDEX_CHUNK_BYTES, max(byte_length, 1)),),
            dtype="uint8",
            overwrite=True,
            compressors=get_compressors(
                profile,
                zarrFormat=_group_zarr_format(group),
            ),
            attributes={
                "byte_length": byte_length,
                "ann_index_format_version": ANN_INDEX_FORMAT_VERSION,
                "metric": str(metric),
                "dimensions": int(dimensions),
                "element_count": int(element_count),
                "payload_sha256": digest.hexdigest(),
            },
        )
        with open(path, "rb") as source:
            for start in range(0, byte_length, ANN_INDEX_IO_BYTES):
                values = np.frombuffer(
                    source.read(min(ANN_INDEX_IO_BYTES, byte_length - start)),
                    dtype=np.uint8,
                )
                array[start : start + len(values)] = values
    finally:
        os.unlink(path)


def validate_ann_index_contract(
    group: zarr.Group,
    space: str,
    dim: int,
    expected_count: int | None = None,
    name: str = ANN_INDEX_ARRAY,
) -> _ValidatedAnnIndexPayload:
    """Validate ANN payload geometry and its complete metadata record."""
    if name not in group:
        raise FileNotFoundError(f"ANN index array {name!r} not found in group")
    source = as_zarr_array(group[name], name=name)
    if (
        source.ndim != 1
        or int(source.shape[0]) < 1
        or np.dtype(source.dtype) != np.dtype(np.uint8)
    ):
        raise ValueError("ANN index payload must be a one-dimensional uint8 array")
    missing = [key for key in _ANN_INDEX_METADATA if key not in source.attrs]
    if missing:
        raise ValueError(f"ANN index metadata is missing: {', '.join(missing)}")
    attrs = source.attrs
    if not _is_int(attrs["byte_length"]) or attrs["byte_length"] != int(
        source.shape[0]
    ):
        raise ValueError("ANN index byte length does not match its payload")
    if attrs["ann_index_format_version"] != ANN_INDEX_FORMAT_VERSION or not _is_int(
        attrs["ann_index_format_version"]
    ):
        raise ValueError("ANN index format version is unsupported")
    if not isinstance(attrs["metric"], str) or attrs["metric"] != space:
        raise ValueError("ANN index metric does not match artifact provenance")
    if not _is_int(attrs["dimensions"]) or attrs["dimensions"] != dim:
        raise ValueError("ANN index dimensions do not match artifact provenance")
    stored_count = attrs["element_count"]
    if isinstance(stored_count, bool) or not isinstance(stored_count, int):
        raise ValueError("ANN index element count is invalid")
    if expected_count is not None and stored_count != int(expected_count):
        raise ValueError("ANN index element count does not match coordinates")
    stored_digest = attrs["payload_sha256"]
    if not isinstance(stored_digest, str):
        raise ValueError("ANN index payload digest is invalid")
    return _ValidatedAnnIndexPayload(
        source=source,
        stored_count=stored_count,
        stored_digest=stored_digest,
    )


def validate_ann_index_payload(
    group: zarr.Group,
    space: str,
    dim: int,
    expected_count: int | None = None,
    name: str = ANN_INDEX_ARRAY,
) -> None:
    """Validate ANN bytes and metadata without instantiating hnswlib."""
    validated = validate_ann_index_contract(group, space, dim, expected_count, name)
    digest = hashlib.sha256()
    source = validated.source
    for start in range(0, int(source.shape[0]), ANN_INDEX_CHUNK_BYTES):
        stop = min(start + ANN_INDEX_CHUNK_BYTES, int(source.shape[0]))
        digest.update(np.asarray(source[start:stop], dtype=np.uint8))
    if digest.hexdigest() != validated.stored_digest:
        raise ValueError("ANN index payload digest does not match its metadata")


def load_ann_index(
    group: zarr.Group,
    space: str,
    dim: int,
    expected_count: int | None = None,
    name: str = ANN_INDEX_ARRAY,
) -> Any:
    """Load an hnswlib index from a Zarr byte array."""
    import hnswlib

    validated = validate_ann_index_contract(group, space, dim, expected_count, name)
    source = validated.source
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        digest = hashlib.sha256()
        with open(path, "wb") as destination:
            for start in range(0, int(source.shape[0]), ANN_INDEX_IO_BYTES):
                values = np.asarray(
                    source[
                        start : min(
                            start + ANN_INDEX_IO_BYTES,
                            int(source.shape[0]),
                        )
                    ],
                    dtype=np.uint8,
                )
                digest.update(values)
                destination.write(memoryview(values))
                # Release the window before the next read allocates its own.
                del values
        if digest.hexdigest() != validated.stored_digest:
            raise ValueError("ANN index payload digest does not match its metadata")
        index = hnswlib.Index(space=space, dim=dim)
        index.load_index(path)
        if int(index.get_current_count()) != validated.stored_count:
            raise ValueError("ANN index element count does not match coordinates")
        return index
    finally:
        os.unlink(path)
