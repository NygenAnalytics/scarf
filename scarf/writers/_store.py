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
from ..storage.count_matrix import CountMatrixPolicy
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
        g: Parent Zarr group.
        name: Array name within the group.
        chunks: Chunk shape, or a single integer applied to every axis.
        dtype: NumPy dtype for the array.
        shape: Array shape.
        overwrite: If True, replace an existing array of the same name.

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
    """Creates and returns a metadata column array.

    Args:
        g: Parent Zarr group.
        name: Array name within the group.
        data: Values to write, or None to create an empty array.
        dtype: Optional dtype. Inferred from ``data`` when omitted.
        overwrite: If True, replace an existing array of the same name.
        chunk_size: Chunk length along the first axis.
        shape: Explicit length when ``data`` is None.

    Returns:
        A Zarr Array.
    """
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
    policy: CountMatrixPolicy | None = None,
) -> zarr.Array:
    """Creates and returns a Zarr array with name 'counts'.

    Args:
        z: Root Zarr group.
        assay_name: Assay group that will own the counts array.
        workspace: Workspace name. None uses the legacy layout.
        n_cells: Number of cells (rows).
        feat_ids: Feature identifiers written to feature metadata.
        feat_names: Feature display names written to feature metadata.
        dtype: Storage dtype for counts.
        profile: Zarr encoding profile. When None, chosen from the store.
        policy: Count-matrix geometry policy. When None, the default plan
                is used.

    Returns:
        The created ``counts`` array.
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
        policy=policy,
    )


def load_count_store(
    z: zarr.Group, assay_name: str, workspace: str | None
) -> zarr.Array:
    """Return the counts Zarr array for an assay in the given workspace.

    Args:
        z: Root Zarr group.
        assay_name: Assay that owns the counts array.
        workspace: Workspace name. None uses the legacy layout.

    Returns:
        The ``counts`` array.
    """
    return _load_count_array(z, assay_name, workspace)


def create_cell_data(
    z: zarr.Group,
    workspace: str | None,
    ids: np.ndarray,
    names: np.ndarray,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    """Create the cellData group with ids, names, and default ``I`` filter column.

    Args:
        z: Root Zarr group.
        workspace: Workspace name. None uses the legacy layout.
        ids: Cell identifiers.
        names: Cell display names.
        profile: Zarr encoding profile. When None, chosen from the store.

    Returns:
        The created ``cellData`` group.
    """
    return _create_cell_data(z, workspace, ids, names, profile=profile)
