from collections.abc import Iterable
from dataclasses import dataclass
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

_MISSING_COLUMN_PREFIX = "__scarf_missing__"


@dataclass(frozen=True, slots=True)
class MetadataBlock:
    """A contiguous block for one metadata column."""

    start: int
    values: np.ndarray
    missing: np.ndarray | None = None


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


def create_streamed_metadata_column(
    group: zarr.Group,
    name: str,
    *,
    shape: int,
    dtype: Any,
    blocks: Iterable[MetadataBlock],
    overwrite: bool = True,
    chunkSize: int = 100_000,
    hasMissing: bool = False,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Create and fill a metadata column from bounded contiguous blocks."""
    if shape < 0:
        raise ValueError("shape must be non-negative")
    if chunkSize < 1:
        raise ValueError("chunkSize must be positive")
    output = create_metadata_column(
        group,
        name,
        dtype=dtype,
        overwrite=overwrite,
        chunkSize=chunkSize,
        shape=shape,
        profile=profile,
    )
    missing_output: zarr.Array | None = None
    if hasMissing:
        missing_name = f"{_MISSING_COLUMN_PREFIX}{name}"
        missing_output = create_metadata_column(
            group,
            missing_name,
            dtype=bool,
            overwrite=overwrite,
            chunkSize=chunkSize,
            shape=shape,
            profile=profile,
        )
        output.attrs["missing_mask"] = missing_name

    next_row = 0
    for block in blocks:
        if block.start != next_row:
            raise ValueError(
                f"Metadata blocks must be contiguous; expected {next_row}, "
                f"received {block.start}"
            )
        values = np.asarray(block.values)
        if values.ndim != 1:
            raise ValueError("Metadata blocks must be one-dimensional")
        stop = block.start + len(values)
        if stop > shape:
            raise ValueError("Metadata block exceeds declared shape")
        if values.dtype != np.dtype(dtype):
            values = values.astype(dtype)
        output[block.start : stop] = values

        if block.missing is not None:
            if missing_output is None:
                raise ValueError("A missing mask was supplied but hasMissing is false")
            missing = np.asarray(block.missing, dtype=bool)
            if missing.shape != values.shape:
                raise ValueError("Missing mask must align with metadata values")
            missing_output[block.start : stop] = missing
        elif missing_output is not None:
            missing_output[block.start : stop] = False
        next_row = stop

    if next_row != shape:
        raise ValueError(
            f"Metadata column is incomplete: wrote {next_row} of {shape} rows"
        )
    return output


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
