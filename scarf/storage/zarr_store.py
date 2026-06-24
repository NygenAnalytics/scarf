import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

StorageProfile = Literal["fast_local", "cloud"]

PROFILE_COUNT_CHUNKS: dict[StorageProfile, tuple[int, int]] = {
    "fast_local": (512, 512),
    "cloud": (256, 256),
}
PROFILE_COUNT_SHARDS: dict[StorageProfile, tuple[int, int]] = {
    "fast_local": (4096, 4096),
    "cloud": (8192, 8192),
}
PROFILE_METADATA_CHUNK = 100_000


@dataclass
class ZarrArraySpec:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: Any
    shards: tuple[int, ...] | None = None
    zarrFormat: int = 3
    compressors: list | None = None
    fillValue: Any | None = None
    overwrite: bool = True


def get_compressors(profile: StorageProfile = "fast_local", zarrFormat: int = 3) -> list:
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


def zarr_root_path(group: zarr.Group) -> str | None:
    store = group.store
    root = getattr(store, "root", None)
    if root is not None:
        return root
    storePath = getattr(group, "store_path", None)
    if storePath and str(storePath).startswith("file://"):
        return str(storePath)[7:]
    return None


def normalize_chunks(chunks: tuple[int, ...] | int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(chunks, int):
        chunks = (chunks,)
    if len(chunks) == len(shape):
        return tuple(min(chunk, dim) for chunk, dim in zip(chunks, shape))
    if len(chunks) == 1 and len(shape) > 1:
        return (min(chunks[0], shape[0]),) + tuple(shape[1:])
    if len(chunks) == 1:
        return (min(chunks[0], shape[0]),)
    raise ValueError(
        f"Cannot map chunks {chunks} to array shape {shape}. "
        "Provide one chunk size per dimension."
    )


_activeProfile: StorageProfile | None = None


def set_storage_profile(profile: StorageProfile | None) -> None:
    global _activeProfile
    _activeProfile = profile


def get_storage_profile() -> StorageProfile:
    if _activeProfile is not None:
        return _activeProfile
    envProfile = os.environ.get("SCARF_ZARR_PROFILE", "fast_local")
    if envProfile in ("fast_local", "cloud"):
        return envProfile  # type: ignore[return-value]
    return "fast_local"


def count_array_spec(
    nCells: int,
    nFeats: int,
    dtype: Any = "uint32",
    profile: StorageProfile | None = None,
    sharded: bool = False,
) -> ZarrArraySpec:
    profile = profile or get_storage_profile()
    chunks = PROFILE_COUNT_CHUNKS[profile]
    shards = PROFILE_COUNT_SHARDS[profile] if sharded else None
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=get_compressors(profile),
        fillValue=0,
    )


def metadata_array_spec(
    length: int,
    dtype: Any,
    profile: StorageProfile | None = None,
) -> ZarrArraySpec:
    chunkSize = min(PROFILE_METADATA_CHUNK, max(length, 1))
    return ZarrArraySpec(
        shape=(length,),
        chunks=(chunkSize,),
        dtype=dtype,
        compressors=get_compressors(profile or get_storage_profile()),
    )


def open_store(path: str, mode: str = "r") -> zarr.Group:
    return zarr.open_group(path, mode=mode)


def create_numeric_array(
    group: zarr.Group,
    name: str,
    spec: ZarrArraySpec,
) -> zarr.Array:
    zarrFormat = spec.zarrFormat if spec.zarrFormat != 3 else _group_zarr_format(group)
    chunks = normalize_chunks(spec.chunks, spec.shape)
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": chunks,
        "dtype": spec.dtype,
        "compressors": spec.compressors or get_compressors(zarrFormat=zarrFormat),
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None:
        kwargs["shards"] = spec.shards
    if spec.fillValue is not None:
        kwargs["fill_value"] = spec.fillValue
    return group.create_array(name, **kwargs)


def dtype_fix(dtype, data: np.ndarray):
    if dtype is None or dtype == object:
        return "U" + str(max([len(str(x)) for x in data]))
    if np.issubdtype(data.dtype, np.dtype("S")):
        try:
            adata = data.astype("U")
        except UnicodeDecodeError:
            adata = np.array([x.decode("UTF-8") for x in data]).astype("U")
        return adata.dtype
    return dtype


def create_metadata_column(
    group: zarr.Group,
    name: str,
    data: np.ndarray | list | None = None,
    dtype: Any = None,
    overwrite: bool = True,
    chunkSize: int | None = None,
    shape: int | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    if chunkSize is None or chunkSize is False:
        chunks: tuple[int, ...] | bool = False
    else:
        chunks = (chunkSize,)

    compressors = get_compressors(
        profile or get_storage_profile(),
        zarrFormat=_group_zarr_format(group),
    )

    if data is not None:
        data = np.array(data)
        data = np.asarray(data, dtype=dtype_fix(dtype, data))
        if chunks is False:
            chunks = (len(data),)
        return group.create_array(
            name,
            data=data,
            chunks=chunks,
            overwrite=overwrite,
            compressors=compressors,
        )

    if shape is None:
        raise ValueError("shape is required when data is None")
    if chunks is False:
        chunks = (shape,)
    return group.create_array(
        name,
        shape=(shape,),
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        compressors=compressors,
    )


def repack_to_sharded(
    srcArray: zarr.Array,
    dstGroup: zarr.Group,
    name: str,
    shards: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    profile: StorageProfile | None = None,
    overwrite: bool = True,
) -> zarr.Array:
    chunks = chunks or tuple(srcArray.chunks)
    spec = ZarrArraySpec(
        shape=srcArray.shape,
        chunks=chunks,
        shards=shards,
        dtype=srcArray.dtype,
        compressors=get_compressors(profile or get_storage_profile()),
        overwrite=overwrite,
    )
    dstArray = create_numeric_array(dstGroup, name, spec)
    shardRows, shardCols = shards[0], shards[1]
    nRows, nCols = srcArray.shape
    for rowStart in range(0, nRows, shardRows):
        rowEnd = min(rowStart + shardRows, nRows)
        for colStart in range(0, nCols, shardCols):
            colEnd = min(colStart + shardCols, nCols)
            dstArray[rowStart:rowEnd, colStart:colEnd] = srcArray[
                rowStart:rowEnd, colStart:colEnd
            ]
    return dstArray


def finalize_sharded_counts(
    store: zarr.Group,
    assayName: str,
    workspace: str | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    profile = profile or get_storage_profile()
    if workspace is None:
        countsPath = f"{assayName}/counts"
        assayGroup = store[assayName]
    else:
        countsPath = f"matrices/{assayName}/counts"
        assayGroup = store[f"matrices/{assayName}"]

    srcArray = store[countsPath]
    if srcArray.metadata.shards is not None:
        return srcArray
    if _group_zarr_format(store) == 2:
        return srcArray

    shards = tuple(
        min(s, d) for s, d in zip(PROFILE_COUNT_SHARDS[profile], srcArray.shape)
    )
    chunks = tuple(
        min(c, d) for c, d in zip(PROFILE_COUNT_CHUNKS[profile], srcArray.shape)
    )
    normalized: list[tuple[int, int]] = []
    for shardDim, chunkDim in zip(shards, chunks):
        if shardDim <= chunkDim or shardDim % chunkDim != 0:
            normalized.append((shardDim, shardDim))
        else:
            normalized.append((shardDim, chunkDim))
    shards = tuple(dim[0] for dim in normalized)
    chunks = tuple(dim[1] for dim in normalized)
    if shards == srcArray.shape:
        return srcArray
    tmpName = "counts__sharded_tmp"
    if tmpName in assayGroup:
        del assayGroup[tmpName]
    repack_to_sharded(
        srcArray,
        assayGroup,
        tmpName,
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup["counts"]
    repack_to_sharded(
        assayGroup[tmpName],
        assayGroup,
        "counts",
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup[tmpName]

    if workspace is None:
        assayGroup = store[assayName]
    else:
        assayGroup = store[f"matrices/{assayName}"]
    assayGroup.attrs["scarf:zarr_spec"] = {
        "profile": profile,
        "chunks": list(chunks),
        "shards": list(shards),
        "zarr_format": 3,
    }
    return store[countsPath]


def array_info(array: zarr.Array) -> str:
    metadata = array.metadata
    parts = [f"shape={array.shape}", f"dtype={array.dtype}", f"chunks={array.chunks}"]
    if metadata.shards is not None:
        parts.append(f"shards={metadata.shards}")
    return ", ".join(parts)
