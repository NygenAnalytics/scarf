import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import zarr

from .types import as_zarr_array
from .arrays import (
    _decode_metadata_values,
    create_metadata_column,
    create_numeric_array,
    dtype_fix,
    MetadataBlock,
    create_streamed_metadata_column,
)
from .budget import ResourceBudget
from .geometry import array_geometry
from .layout import PROFILE_METADATA_CHUNK, normed_array_spec
from .partition import row_band
from .profiles import StorageProfile
from .sharding import write_dense_in_shard_rows

COLUMN_METADATA_ATTRIBUTES = (
    "display",
    "feature_selection_fingerprint",
    "levels",
    "ordered",
    "role",
    "assay",
    "description",
    "unit",
)


def copy_zarr_array(
    src: zarr.Array,
    dst: zarr.Array,
    msg: str | None = None,
    resources: ResourceBudget | None = None,
) -> None:
    """Stream-copy a 2D Zarr array in row blocks."""
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")
    if len(src.shape) != 2:
        raise ValueError("copy_zarr_array only supports 2D arrays")
    write_dense_in_shard_rows(
        dst,
        lambda start, end: np.asarray(src[start:end, :]),
        msg=msg or "Copying Zarr array",
        resources=resources,
        producerBytes=int(np.prod(src.chunks)) * src.dtype.itemsize,
    )


def _metadata_block_rows(array: zarr.Array) -> int:
    return row_band(
        array_geometry(array),
        unit="chunk",
        fallback=PROFILE_METADATA_CHUNK,
    )


def _resolve_metadata_dtype(
    array: zarr.Array,
    block_rows: int,
) -> np.dtype[Any]:
    dtype: np.dtype[Any] = np.dtype(array.dtype)
    if not (dtype.kind in {"O", "S"} or dtype.hasobject):
        return dtype
    if dtype.kind == "S":
        return np.empty(0, dtype=dtype).astype(str).dtype
    n_rows = int(array.shape[0])
    if n_rows == 0:
        return np.dtype("U1")
    max_len = 1
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        block = np.asarray(array[start:stop])
        if block.size == 0:
            continue
        resolved: np.dtype[Any] = np.dtype(dtype_fix(dtype, block))
        if resolved.kind == "U":
            max_len = max(max_len, resolved.itemsize // 4)
        else:
            return resolved
    return np.dtype(f"U{max_len}")


def _copy_metadata_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    overwrite: bool,
    profile: StorageProfile | None = None,
    row_indices: np.ndarray | None = None,
    missing: zarr.Array | None = None,
) -> None:
    if src.ndim != 1:
        create_metadata_column(
            dst,
            name,
            data=np.asarray(src[:]),
            dtype=src.dtype,
            overwrite=overwrite,
            chunkSize=PROFILE_METADATA_CHUNK,
            profile=profile,
        )
        target = as_zarr_array(dst[name], name=name)
    else:
        block_rows = _metadata_block_rows(src)
        dtype = _resolve_metadata_dtype(src, block_rows)
        n_rows = int(src.shape[0]) if row_indices is None else len(row_indices)

        def blocks() -> Iterator[MetadataBlock]:
            for start in range(0, n_rows, block_rows):
                stop = min(start + block_rows, n_rows)
                rows = (
                    slice(start, stop)
                    if row_indices is None
                    else row_indices[start:stop]
                )
                values = src[rows] if isinstance(rows, slice) else src.oindex[rows]
                mask = (
                    None
                    if missing is None
                    else (
                        missing[rows]
                        if isinstance(rows, slice)
                        else missing.oindex[rows]
                    )
                )
                yield MetadataBlock(
                    start=start,
                    values=np.asarray(_decode_metadata_values(values), dtype=dtype),
                    missing=None if mask is None else np.asarray(mask, dtype=bool),
                )

        target = create_streamed_metadata_column(
            dst,
            name,
            dtype=dtype,
            overwrite=overwrite,
            chunkSize=PROFILE_METADATA_CHUNK,
            shape=n_rows,
            profile=profile,
            blocks=blocks(),
            hasMissing=missing is not None,
        )
    for attribute in COLUMN_METADATA_ATTRIBUTES:
        if attribute in src.attrs:
            target.attrs[attribute] = src.attrs[attribute]


def copy_metadata_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    overwrite: bool = True,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Stream-copy one metadata vector without carrying presentation attrs."""
    _copy_metadata_array(
        src,
        dst,
        name,
        overwrite=overwrite,
        profile=profile,
    )
    target = as_zarr_array(dst[name], name=name)
    for attribute in tuple(target.attrs):
        del target.attrs[attribute]
    return target


def copy_zarr_group_tree(
    src: zarr.Group,
    dst: zarr.Group,
    *,
    overwrite: bool = True,
    exclude_members: set[str] | frozenset[str] | None = None,
    row_indices: np.ndarray | None = None,
    profile: StorageProfile | None = None,
) -> None:
    """Recursively copy a Zarr group tree.

    ``exclude_members`` applies only to immediate members of ``src``. Child
    groups are copied recursively without inheriting the parent exclusions.
    """
    masks = validate_metadata_dependencies(src, exclude_members=exclude_members)
    hidden = {
        link
        for _, array in src.arrays()
        if isinstance(link := array.attrs.get("missing_mask"), str) and "/" not in link
    }
    for name, node in src.members():
        if exclude_members is not None and name in exclude_members:
            continue
        if name in hidden or name.startswith("__scarf_missing__"):
            continue
        if isinstance(node, zarr.Group):
            child = dst.create_group(name, overwrite=overwrite)
            copy_zarr_group_tree(
                node,
                child,
                overwrite=overwrite,
                row_indices=row_indices,
                profile=profile,
            )
        else:
            array = as_zarr_array(node, name=name)
            _copy_metadata_array(
                array,
                dst,
                name,
                overwrite=overwrite,
                row_indices=row_indices,
                missing=masks.get(name),
                profile=profile,
            )


def validate_metadata_dependencies(
    group: zarr.Group,
    *,
    exclude_members: set[str] | frozenset[str] | None = None,
) -> dict[str, zarr.Array]:
    masks = {}
    for name, array in group.arrays():
        if exclude_members is not None and name in exclude_members:
            continue
        link = array.attrs.get("missing_mask")
        if link is None:
            continue
        if (
            not isinstance(link, str)
            or "/" in link
            or link == name
            or link not in group
        ):
            raise ValueError(
                f"Column {name!r} has a missing or malformed missing-value dependency"
            )
        mask = as_zarr_array(group[link], name=link)
        if (
            mask.dtype != np.dtype(bool)
            or mask.shape != array.shape
            or mask.attrs.get("missing_mask") is not None
        ):
            raise ValueError(f"Column {name!r} has a malformed missing-value array")
        masks[name] = mask
    return masks


def create_or_open_staged_normed_array(
    cache_path: str,
    shape: tuple[int, int],
) -> zarr.Array:
    """Open or create a reusable local normalized-data array."""
    if os.path.exists(os.path.join(cache_path, "zarr.json")):
        root = zarr.open_group(cache_path, mode="r+")
        if "data" in root:
            array = as_zarr_array(root["data"], name="data")
            if tuple(array.shape) == tuple(shape):
                return array
    root = zarr.open_group(cache_path, mode="w")
    spec = normed_array_spec(shape[0], shape[1], profile="fast_local")
    return create_numeric_array(root, "data", spec)
