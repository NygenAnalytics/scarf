from collections.abc import Callable, Iterable, Mapping
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
from .types import as_zarr_array

MISSING_MASK_PREFIX = "__scarf_missing__"
"""Name prefix of the boolean array that flags missing values of a column."""


@dataclass(frozen=True, slots=True)
class MetadataBlock:
    """A contiguous block for one metadata column."""

    start: int
    values: np.ndarray
    missing: np.ndarray | None = None


def linked_missing_mask(
    group: zarr.Group,
    name: str,
    *,
    label: str | None = None,
    values: zarr.Array | None = None,
) -> zarr.Array | None:
    """Return the missing-value mask linked to ``group[name]``, if it has one.

    A nullable array names its mask in ``attrs["missing_mask"]``. The link must
    name the canonical ``__scarf_missing__<name>`` sibling, and that sibling
    must be a boolean array with the same shape. ``label`` names the array in
    errors and defaults to ``Array '<name>'``. Pass the already opened
    ``values`` array to skip reopening it.
    """
    if values is None:
        values = as_zarr_array(group[name], name=name)
    if "missing_mask" not in values.attrs:
        return None
    subject = f"Array {name!r}" if label is None else label
    missing_name = f"{MISSING_MASK_PREFIX}{name}"
    if values.attrs["missing_mask"] != missing_name:
        raise ValueError(f"{subject} has a malformed missing-mask link")
    try:
        mask = group[missing_name]
    except KeyError:
        raise ValueError(f"{subject} has a missing missing-mask array") from None
    if (
        not isinstance(mask, zarr.Array)
        or mask.dtype != np.dtype(bool)
        or mask.shape != values.shape
        or "missing_mask" in mask.attrs
    ):
        raise ValueError(f"{subject} has a malformed missing-mask array")
    return mask


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


def text_value(value: Any) -> str:
    """Return one metadata value as text, decoding UTF-8 bytes and None as empty."""
    if isinstance(value, bytes | bytearray | np.bytes_):
        return bytes(value).decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _decode_metadata_values(data: Any) -> np.ndarray:
    """Decode UTF-8 byte strings and missing values, keeping the array shape."""
    values = np.asarray(data)
    if values.dtype.kind == "S" or (
        values.dtype.hasobject
        and any(
            value is None or isinstance(value, bytes | bytearray | np.bytes_)
            for value in values.flat
        )
    ):
        return np.asarray(
            [text_value(value) for value in values.flat], dtype=str
        ).reshape(values.shape)
    return values


def _measured_text_dtype(blocks: Iterable[Any]) -> np.dtype[Any]:
    width = 1
    for block in blocks:
        for value in np.asarray(block).flat:
            width = max(width, len(text_value(value)))
    return np.dtype(f"U{width}")


def text_dtype(dtype: Any, blocks: Callable[[], Iterable[Any]]) -> np.dtype[Any]:
    """Return the fixed-width unicode dtype that holds values as decoded text.

    Fixed-width strings keep their declared width, because UTF-8 decoding never
    adds characters, and other fixed-width types use NumPy's text width. Object
    and variable-width string values are measured from the ``blocks`` callable.
    """
    resolved: np.dtype[Any] = np.dtype(dtype)
    if not resolved.hasobject:
        return np.empty(0, dtype=resolved).astype(str).dtype
    return _measured_text_dtype(blocks())


def stored_metadata_dtype(
    dtype: Any, blocks: Callable[[], Iterable[Any]]
) -> np.dtype[Any]:
    """Return the dtype a copied column is stored with: text as fixed-width unicode."""
    resolved: np.dtype[Any] = np.dtype(dtype)
    if resolved.kind in {"S", "U"} or resolved.hasobject:
        return text_dtype(resolved, blocks)
    return resolved


def encode_metadata_values(
    data: Any,
    dtype: Any = None,
    *,
    name: str,
) -> np.ndarray:
    """Return metadata values in the form that a metadata column stores."""
    raw = np.asarray(data)
    try:
        values = _decode_metadata_values(raw)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Column {name!r} holds text that is not valid UTF-8; pass str values "
            "or UTF-8 encoded bytes"
        ) from exc
    # Text is stored at its decoded width, whatever the source S or O dtype.
    if (
        dtype is None
        or np.dtype(dtype).kind == "O"
        or raw.dtype.kind in {"S", "O"}
        and values.dtype.kind == "U"
    ):
        return values.astype(_measured_text_dtype((values,)))
    return np.asarray(values, dtype=dtype)


def create_metadata_column(
    group: zarr.Group,
    name: str,
    data: np.ndarray | list[Any] | None = None,
    dtype: Any = None,
    overwrite: bool = True,
    chunkSize: int | bool | None = None,
    shape: int | None = None,
    profile: StorageProfile | None = None,
    *,
    attributes: Mapping[str, Any] | None = None,
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
    attrs = None if attributes is None else dict(attributes)

    if data is not None:
        values = encode_metadata_values(data, dtype, name=name)
        if chunks is False:
            chunks = (max(1, len(values)),)
        return group.create_array(
            name,
            data=values,
            chunks=chunks,
            overwrite=overwrite,
            compressors=compressors,
            attributes=attrs,
        )

    if shape is None:
        raise ValueError("shape is required when data is None")
    if chunks is False:
        chunks = (max(1, shape),)
    return group.create_array(
        name,
        shape=(shape,),
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        compressors=compressors,
        attributes=attrs,
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
    attributes: Mapping[str, Any] | None = None,
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
        attributes=attributes,
    )
    missing_output: zarr.Array | None = None
    if hasMissing:
        missing_name = f"{MISSING_MASK_PREFIX}{name}"
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
