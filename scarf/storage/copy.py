from collections.abc import Iterator
from pathlib import Path

import numpy as np
import zarr

from .types import as_zarr_array
from .arrays import (
    MISSING_MASK_PREFIX,
    _decode_metadata_values,
    create_numeric_array,
    linked_missing_mask,
    MetadataBlock,
    create_streamed_metadata_column,
    stored_metadata_dtype,
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


def _copy_metadata_array(
    src: zarr.Array,
    dst: zarr.Group,
    name: str,
    *,
    overwrite: bool,
    profile: StorageProfile | None = None,
    row_indices: np.ndarray | None = None,
    missing: zarr.Array | None = None,
    copy_attributes: bool = True,
) -> None:
    """Stream-copy one metadata column, its rows ``row_indices`` or all.

    With ``copy_attributes``, the attributes in ``COLUMN_METADATA_ATTRIBUTES``
    that ``src`` carries, such as the role and assay of a membership column,
    are part of the copy's first metadata write.
    """
    if src.ndim != 1:
        raise ValueError(
            f"Metadata column {name!r} has {src.ndim} dimensions; metadata columns "
            "must be one-dimensional"
        )
    block_rows = _metadata_block_rows(src)
    n_source = int(src.shape[0])
    dtype = stored_metadata_dtype(
        src.dtype,
        lambda: (
            src[start : start + block_rows] for start in range(0, n_source, block_rows)
        ),
    )
    n_rows = n_source if row_indices is None else len(row_indices)

    def blocks() -> Iterator[MetadataBlock]:
        for start in range(0, n_rows, block_rows):
            stop = min(start + block_rows, n_rows)
            rows = (
                slice(start, stop) if row_indices is None else row_indices[start:stop]
            )
            values = src[rows] if isinstance(rows, slice) else src.oindex[rows]
            mask = (
                None
                if missing is None
                else (
                    missing[rows] if isinstance(rows, slice) else missing.oindex[rows]
                )
            )
            yield MetadataBlock(
                start=start,
                values=np.asarray(_decode_metadata_values(values)).astype(dtype),
                missing=None if mask is None else np.asarray(mask, dtype=bool),
            )

    attributes = (
        {key: src.attrs[key] for key in COLUMN_METADATA_ATTRIBUTES if key in src.attrs}
        if copy_attributes
        else {}
    )
    create_streamed_metadata_column(
        dst,
        name,
        dtype=dtype,
        overwrite=overwrite,
        chunkSize=PROFILE_METADATA_CHUNK,
        shape=n_rows,
        profile=profile,
        blocks=blocks(),
        hasMissing=missing is not None,
        attributes=attributes or None,
    )


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
        copy_attributes=False,
    )
    return as_zarr_array(dst[name], name=name)


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
    for name, node in src.members():
        if exclude_members is not None and name in exclude_members:
            continue
        # A mask is copied together with the column that links it.
        if name.startswith(MISSING_MASK_PREFIX):
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
        mask = linked_missing_mask(group, name, label=f"Column {name!r}", values=array)
        if mask is not None:
            masks[name] = mask
    return masks


def create_or_open_staged_normed_array(
    cache_path: str,
    shape: tuple[int, int],
) -> zarr.Array:
    """Open or create a reusable local normalized-data array."""
    path = Path(cache_path)
    if (path / "zarr.json").exists():
        root = zarr.open_group(path, mode="r+")
        if "data" in root:
            array = as_zarr_array(root["data"], name="data")
            if tuple(array.shape) == tuple(shape):
                return array
    root = zarr.open_group(path, mode="w")
    spec = normed_array_spec(shape[0], shape[1], profile="fast_local")
    return create_numeric_array(root, "data", spec)
