from typing import Any

import numpy as np
import zarr

from ..storage.arrays import (
    create_zarr_dataset as _create_zarr_dataset,
    create_zarr_obj_array as _create_zarr_obj_array,
)
from ..storage.schema import (
    create_cell_data as _create_cell_data,
    create_zarr_count_assay as _create_zarr_count_assay,
    load_count_array as _load_count_array,
)
from ..storage.profiles import StorageProfile
from ..storage.stores import load_zarr as load_zarr


def create_zarr_dataset(
    g: zarr.Group,
    name: str,
    chunks: tuple[int, ...] | int,
    dtype: Any,
    shape: tuple[int, ...],
    overwrite: bool = True,
) -> zarr.Array:
    """Creates and returns a Zarr array.

    Args:
        g (zarr.hierarchy):
        name (str):
        chunks (tuple):
        dtype (Any):
        shape (Tuple):
        overwrite (bool):

    Returns:
        A Zarr Array.
    """
    return _create_zarr_dataset(g, name, chunks, dtype, shape, overwrite)


def create_zarr_obj_array(
    g: zarr.Group,
    name: str,
    data: Any,
    dtype: str | Any = None,
    overwrite: bool = True,
    chunk_size: int = 100000,
    shape: int | None = None,
) -> zarr.Array:
    """Creates and returns a metadata column array."""
    return _create_zarr_obj_array(
        g,
        name,
        data,
        dtype,
        overwrite,
        chunk_size,
        shape,
    )


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: str = "uint32",
    *,
    profile: StorageProfile | None = None,
    targetChunkBytes: int | None = None,
    targetShardBytes: int | None = None,
) -> zarr.Array:
    """Creates and returns a Zarr array with name 'counts'.

    Args:
        z (zarr.Group):
        assay_name (str):
        workspace (str | None):
        n_cells (int):
        feat_ids (np.ndarray | list[str]):
        feat_names (np.ndarray | list[str]):
        dtype (str = 'uint32'):
        targetChunkBytes: Optional inner-chunk byte target.
        targetShardBytes: Optional full-width shard byte target.

    Returns:
        A Zarr array.
    """
    return _create_zarr_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        feat_ids,
        feat_names,
        dtype,
        profile=profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
    )


def load_count_store(
    z: zarr.Group, assay_name: str, workspace: str | None
) -> zarr.Array:
    """Return the counts Zarr array for an assay in the given workspace."""
    return _load_count_array(z, assay_name, workspace)


def create_cell_data(
    z: zarr.Group,
    workspace: str | None,
    ids: np.ndarray,
    names: np.ndarray,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    """Create the cellData group with ids, names, and default ``I`` filter column."""
    return _create_cell_data(z, workspace, ids, names, profile=profile)
