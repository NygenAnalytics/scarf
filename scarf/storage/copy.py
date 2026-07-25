import os

import numpy as np
import zarr

from .types import as_zarr_array
from .arrays import create_metadata_column, create_numeric_array
from .layout import array_shard_rows, normed_array_spec
from .sharding import write_dense_in_shard_rows


def copy_zarr_array(
    src: zarr.Array,
    dst: zarr.Array,
    block_rows: int | None = None,
    msg: str | None = None,
) -> None:
    """Stream-copy a 2D Zarr array in row blocks."""
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")
    if len(src.shape) != 2:
        raise ValueError("copy_zarr_array only supports 2D arrays")
    if block_rows is None:
        block_rows = array_shard_rows(dst)
    write_dense_in_shard_rows(
        dst,
        lambda start, end: np.asarray(src[start:end, :]),
        msg=msg or "Copying Zarr array",
        shard_rows=block_rows,
    )


def copy_zarr_group_tree(
    src: zarr.Group,
    dst: zarr.Group,
    *,
    overwrite: bool = True,
) -> None:
    """Recursively copy a Zarr group tree."""
    for name, node in src.members():
        if isinstance(node, zarr.Group):
            child = dst.create_group(name, overwrite=overwrite)
            copy_zarr_group_tree(node, child, overwrite=overwrite)
        else:
            array = as_zarr_array(node, name=name)
            create_metadata_column(
                dst,
                name,
                data=np.asarray(array[:]),
                dtype=array.dtype,
                overwrite=overwrite,
                chunkSize=100000,
            )


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
