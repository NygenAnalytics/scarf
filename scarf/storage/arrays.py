from typing import Any

import numpy as np
import zarr

from .layout import (
    ZarrArraySpec,
    _group_zarr_format,
    get_compressors,
    normalize_chunks,
)
from .profiles import StorageProfile, resolve_storage_profile


def _checked_shards(
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """Return shard extents that hold a whole number of the resolved chunks."""
    resolved = tuple(max(1, int(value)) for value in shards)
    if len(resolved) != len(chunks):
        raise ValueError(f"Array shards {resolved} do not match chunks {chunks}")
    for shard, chunk in zip(resolved, chunks, strict=True):
        if shard < chunk or shard % chunk:
            raise ValueError(
                f"Array shards {resolved} must hold whole chunks {chunks}; "
                f"shard extent {shard} is not a multiple of chunk extent {chunk}"
            )
    return resolved


def create_numeric_array(
    group: zarr.Group,
    name: str,
    spec: ZarrArraySpec,
) -> zarr.Array:
    """Create a numeric Zarr array from a specification."""
    zarrFormat = _group_zarr_format(group)
    chunks = normalize_chunks(spec.chunks, spec.shape)
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": chunks,
        "dtype": spec.dtype,
        "compressors": (
            spec.compressors
            if zarrFormat >= 3
            else get_compressors(
                resolve_storage_profile(group.store),
                zarrFormat=2,
            )
        ),
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None and zarrFormat >= 3:
        kwargs["shards"] = _checked_shards(spec.shards, chunks)
    if spec.fillValue is not None:
        kwargs["fill_value"] = spec.fillValue
    return group.create_array(name, **kwargs)


def dtype_fix(dtype: Any, data: np.ndarray) -> Any:
    """Infer or adjust a metadata dtype from sample values."""
    if dtype is None or np.dtype(dtype).kind == "O":
        return "U" + str(max(len(str(value)) for value in data))
    if np.issubdtype(data.dtype, np.dtype("S")):
        try:
            decoded = data.astype("U")
        except UnicodeDecodeError:
            decoded = np.array([value.decode("UTF-8") for value in data]).astype("U")
        return decoded.dtype
    return dtype


def create_metadata_column(
    group: zarr.Group,
    name: str,
    data: np.ndarray | list[Any] | None = None,
    dtype: Any = None,
    overwrite: bool = True,
    chunkSize: int | bool | None = None,
    shape: int | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Create a metadata column, optionally from provided data."""
    if chunkSize is None or chunkSize is False:
        chunks: tuple[int, ...] | bool = False
    else:
        chunks = (chunkSize,)

    resolved_profile = profile or resolve_storage_profile(group.store)
    compressors = get_compressors(
        resolved_profile,
        zarrFormat=_group_zarr_format(group),
    )

    if data is not None:
        values = np.array(data)
        values = np.asarray(values, dtype=dtype_fix(dtype, values))
        if chunks is False:
            chunks = (len(values),)
        return group.create_array(
            name,
            data=values,
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


def _normalize_chunks(chunks: tuple[int, ...] | int) -> tuple[int, ...]:
    if isinstance(chunks, int):
        return (chunks,)
    return chunks


def create_zarr_dataset(
    group: zarr.Group,
    name: str,
    chunks: tuple[int, ...] | int,
    dtype: Any,
    shape: tuple[int, ...],
    overwrite: bool = True,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    resolved_profile = profile or resolve_storage_profile(group.store)
    spec = ZarrArraySpec(
        shape=shape,
        chunks=_normalize_chunks(chunks),
        dtype=dtype,
        compressors=get_compressors(
            resolved_profile,
            zarrFormat=_group_zarr_format(group),
        ),
        overwrite=overwrite,
    )
    return create_numeric_array(group, name, spec)


def create_zarr_obj_array(
    group: zarr.Group,
    name: str,
    data: Any,
    dtype: str | Any = None,
    overwrite: bool = True,
    chunk_size: int = 100000,
    shape: int | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    return create_metadata_column(
        group,
        name,
        data=data,
        dtype=dtype,
        overwrite=overwrite,
        chunkSize=chunk_size,
        shape=shape if data is None else None,
        profile=profile,
    )
