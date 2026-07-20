from typing import Any

import numpy as np
import zarr

from .layout import (
    ZarrArraySpec,
    _group_zarr_format,
    get_compressors,
    normalize_chunks,
)
from .profiles import StorageProfile, get_storage_profile


def create_numeric_array(
    group: zarr.Group,
    name: str,
    spec: ZarrArraySpec,
) -> zarr.Array:
    """Create a numeric Zarr array from a specification."""
    zarrFormat = _group_zarr_format(group)
    chunks = normalize_chunks(spec.chunks, spec.shape)
    profile = get_storage_profile()
    compressors = get_compressors(profile, zarrFormat=zarrFormat)
    if spec.compressors is not None and zarrFormat >= 3:
        compressors = spec.compressors
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": chunks,
        "dtype": spec.dtype,
        "compressors": compressors,
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None and zarrFormat >= 3:
        kwargs["shards"] = spec.shards
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

    compressors = get_compressors(
        profile or get_storage_profile(),
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
) -> zarr.Array:
    spec = ZarrArraySpec(
        shape=shape,
        chunks=_normalize_chunks(chunks),
        dtype=dtype,
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
) -> zarr.Array:
    return create_metadata_column(
        group,
        name,
        data=data,
        dtype=dtype,
        overwrite=overwrite,
        chunkSize=chunk_size,
        shape=shape if data is None else None,
    )
