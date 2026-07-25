import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

from .types import array_metadata_shards
from .budget import ResourceBudget, get_resource_budget
from .profiles import StorageProfile, get_storage_profile

PROFILE_METADATA_CHUNK = 100_000
_CODEC_MAX_BYTES = 2_147_483_647
DEFAULT_CLOUD_TARGET_CHUNK_BYTES = 128 * 1024 * 1024
DEFAULT_MIN_FEATURE_CHUNK = 500
DEFAULT_MAX_FEATURE_CHUNK = 10_000


def _ceil_pad(n: int, chunk: int) -> int:
    chunk = max(1, int(chunk))
    n = max(1, int(n))
    return math.ceil(n / chunk) * chunk


def _fit_shard_to_byte_limit(
    row_shard: int,
    feature_chunk: int,
    shard_cols: int,
    n_features: int,
    *,
    itemsize: int,
    max_bytes: int,
) -> tuple[int, int, int]:
    """Shrink shard geometry to fit the codec byte limit."""

    def shard_bytes(rs: int, sc: int) -> int:
        return rs * sc * itemsize

    while shard_bytes(row_shard, shard_cols) > max_bytes:
        if row_shard > 1:
            row_shard = max(1, max_bytes // (shard_cols * itemsize))
            while row_shard > 1 and shard_bytes(row_shard, shard_cols) > max_bytes:
                row_shard -= 1
        elif shard_cols > feature_chunk:
            shard_cols = max(
                feature_chunk,
                (max_bytes // (row_shard * itemsize) // feature_chunk) * feature_chunk,
            )
            if shard_bytes(row_shard, shard_cols) > max_bytes:
                shard_cols = max(feature_chunk, shard_cols - feature_chunk)
        elif feature_chunk > 1:
            feature_chunk = max(1, max_bytes // (row_shard * itemsize))
            shard_cols = _ceil_pad(n_features, feature_chunk)
            if shard_bytes(row_shard, shard_cols) > max_bytes:
                shard_cols = max(
                    feature_chunk,
                    (max_bytes // (row_shard * itemsize) // feature_chunk)
                    * feature_chunk,
                )
        else:
            feature_chunk = max(1, max_bytes // itemsize)
            shard_cols = feature_chunk
            row_shard = 1
            break
    return row_shard, feature_chunk, shard_cols


def matrix_layout(
    n_cells: int,
    n_features: int,
    *,
    budget: ResourceBudget | None = None,
    itemsize: int = 4,
    width: int | None = None,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int = 1,
    maxFeatureChunk: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Return chunk and shard geometry from the memory-first layout rule."""
    budget = budget or get_resource_budget()
    n_cells = max(int(n_cells), 1)
    n_features = max(int(n_features), 1)
    itemsize = max(1, int(itemsize))
    work = budget.memoryBytes // max(1, budget.workingCopies)

    if width is not None:
        normalized_width = max(1, int(width))
        row_bytes = normalized_width * itemsize
        if row_bytes > _CODEC_MAX_BYTES:
            raise ValueError(
                f"One full-width row requires {row_bytes} bytes, exceeding "
                f"the codec limit of {_CODEC_MAX_BYTES} bytes"
            )
        max_chunk_bytes = min(work, _CODEC_MAX_BYTES)
        row_shard = max(1, min(n_cells, max_chunk_bytes // row_bytes))
        return (row_shard, normalized_width), None

    row_shard = max(1, min(n_cells, work // (n_features * itemsize)))
    feature_chunk = max(1, min(n_features, work // (n_cells * itemsize)))
    if targetChunkBytes is not None:
        target = max(itemsize, int(targetChunkBytes))
        feature_chunk = max(1, target // (row_shard * itemsize))
        feature_chunk = min(n_features, feature_chunk)
        lower = max(1, int(minFeatureChunk))
        upper = n_features if maxFeatureChunk is None else max(1, int(maxFeatureChunk))
        if lower > upper:
            lower, upper = upper, lower
        feature_chunk = max(lower, min(upper, feature_chunk))
        feature_chunk = min(n_features, feature_chunk)
    shard_cols = _ceil_pad(n_features, feature_chunk)
    max_shard_bytes = min(work, _CODEC_MAX_BYTES)
    row_shard, feature_chunk, shard_cols = _fit_shard_to_byte_limit(
        row_shard,
        feature_chunk,
        shard_cols,
        n_features,
        itemsize=itemsize,
        max_bytes=max_shard_bytes,
    )
    return (row_shard, feature_chunk), (row_shard, shard_cols)


@dataclass
class ZarrArraySpec:
    """Specification for creating a numeric Zarr array."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: Any
    shards: tuple[int, ...] | None = None
    zarrFormat: int = 3
    compressors: list[Any] | None = None
    fillValue: Any | None = None
    overwrite: bool = True


def get_compressors(
    profile: StorageProfile = "fast_local",
    zarrFormat: int = 3,
) -> list[Any]:
    """Return codecs for a storage profile and Zarr format."""
    if zarrFormat == 2:
        from numcodecs import Blosc

        return [Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)]
    if profile == "cloud":
        return [ZstdCodec(level=3)]
    return [BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]


def _group_zarr_format(group: zarr.Group) -> int:
    metadata = getattr(group, "metadata", None)
    if metadata is not None and getattr(metadata, "zarr_format", None) is not None:
        return int(metadata.zarr_format)
    return 3


def normalize_chunks(
    chunks: tuple[int, ...] | int,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a chunk specification to the dimensions in ``shape``."""

    def bounded_chunk(chunk: int, dimension: int) -> int:
        return max(1, min(max(1, int(chunk)), max(1, int(dimension))))

    if isinstance(chunks, int):
        chunks = (chunks,)
    if len(chunks) == len(shape):
        return tuple(
            bounded_chunk(chunk, dimension)
            for chunk, dimension in zip(chunks, shape, strict=True)
        )
    if len(chunks) == 1 and len(shape) > 1:
        return (bounded_chunk(chunks[0], shape[0]),) + tuple(
            max(1, int(dimension)) for dimension in shape[1:]
        )
    if len(chunks) == 1:
        return (bounded_chunk(chunks[0], shape[0]),)
    raise ValueError(
        f"Cannot map chunks {chunks} to array shape {shape}. "
        "Provide one chunk size per dimension."
    )


def count_array_spec(
    nCells: int,
    nFeats: int,
    dtype: Any = "uint32",
    profile: StorageProfile | None = None,
    *,
    remote: bool | None = None,
    budget: ResourceBudget | None = None,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int | None = None,
    maxFeatureChunk: int | None = None,
) -> ZarrArraySpec:
    """Build an array specification for an assay count matrix."""
    profile = profile or get_storage_profile()
    budget = budget or get_resource_budget()
    if remote is None:
        remote = profile == "cloud"
    itemsize = int(np.dtype(dtype).itemsize)

    layout_kwargs: dict[str, Any] = {}
    resolved_target = targetChunkBytes
    if resolved_target is None and remote:
        resolved_target = DEFAULT_CLOUD_TARGET_CHUNK_BYTES
    if resolved_target is not None:
        layout_kwargs["targetChunkBytes"] = resolved_target
        layout_kwargs["minFeatureChunk"] = (
            DEFAULT_MIN_FEATURE_CHUNK if minFeatureChunk is None else minFeatureChunk
        )
        layout_kwargs["maxFeatureChunk"] = (
            DEFAULT_MAX_FEATURE_CHUNK if maxFeatureChunk is None else maxFeatureChunk
        )

    chunks, shards = matrix_layout(
        nCells,
        nFeats,
        budget=budget,
        itemsize=itemsize,
        **layout_kwargs,
    )
    assert shards is not None
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=get_compressors("cloud" if remote else profile),
        fillValue=0,
    )


def metadata_array_spec(
    length: int,
    dtype: Any,
    profile: StorageProfile | None = None,
) -> ZarrArraySpec:
    """Build an array specification for a metadata column."""
    chunkSize = min(PROFILE_METADATA_CHUNK, max(length, 1))
    return ZarrArraySpec(
        shape=(length,),
        chunks=(chunkSize,),
        dtype=dtype,
        compressors=get_compressors(profile or get_storage_profile()),
    )


def normed_array_spec(
    nCells: int,
    nFeats: int,
    profile: StorageProfile | None = None,
    *,
    remote: bool | None = None,
    budget: ResourceBudget | None = None,
) -> ZarrArraySpec:
    """Build an array specification for normalized expression data."""
    profile = profile or get_storage_profile()
    budget = budget or get_resource_budget()
    if remote is None:
        remote = profile == "cloud"
    chunks, _ = matrix_layout(
        nCells,
        nFeats,
        budget=budget,
        itemsize=4,
        width=max(nFeats, 1),
    )
    row_chunk, n_cols = chunks
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=(row_chunk, n_cols),
        shards=None,
        dtype="float32",
        compressors=get_compressors("cloud" if remote else profile),
        fillValue=0.0,
    )


def array_shard_rows(array: zarr.Array) -> int:
    """Return the row extent of a shard, chunk, or full array."""
    shards_meta = array_metadata_shards(array)
    if shards_meta is not None and len(shards_meta) > 0:
        return int(shards_meta[0])
    chunks = getattr(array, "chunks", None)
    if chunks is not None and len(chunks) > 0:
        return int(chunks[0])
    return max(int(array.shape[0]), 1)


def iter_shard_row_slices(
    n_rows: int,
    shard_rows: int,
) -> Iterator[tuple[int, int]]:
    """Yield row slices aligned to a shard height."""
    shard_rows = max(1, int(shard_rows))
    n_rows = max(0, int(n_rows))
    for start in range(0, n_rows, shard_rows):
        yield start, min(start + shard_rows, n_rows)


def array_info(array: zarr.Array) -> str:
    """Return a compact description of a Zarr array."""
    parts = [f"shape={array.shape}", f"dtype={array.dtype}", f"chunks={array.chunks}"]
    shards_meta = array_metadata_shards(array)
    if shards_meta is not None:
        parts.append(f"shards={shards_meta}")
    return ", ".join(parts)
