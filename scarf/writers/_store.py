from collections.abc import Iterator
from typing import Any

import numpy as np
import zarr
from scipy.sparse import coo_matrix

from ..storage.arrays import (
    create_zarr_dataset as _create_zarr_dataset,
    create_zarr_obj_array as _create_zarr_obj_array,
)
from ..storage.schema import (
    create_cell_data as _create_cell_data,
    create_zarr_count_assay as _create_zarr_count_assay,
    finalize_counts as _finalize_counts,
    load_count_array as _load_count_array,
)
from ..storage.sharding import accumulate_sparse_to_shards
from ..storage.stores import ZARRLOC as ZARRLOC
from ..storage.stores import load_zarr as load_zarr


def _apply_budget_override(
    mem_budget: int | str | None,
    nthreads: int | None,
    working_copies: int | None,
) -> None:
    """Install a process resource budget from writer-level overrides, if given.

    Write-time chunk and shard geometry is derived from the active resource
    budget. Passing any of these lets a caller simulate writing on a machine
    with a different memory size or core count than the one running the code.
    When all three are None the currently active budget is left untouched, so
    callers that set it themselves keep control.
    """
    if mem_budget is None and nthreads is None and working_copies is None:
        return
    from ..storage.budget import resolve_budget, set_resource_budget

    set_resource_budget(
        resolve_budget(
            memory=mem_budget, workers=nthreads, working_copies=working_copies
        )
    )


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
    chunk_size: tuple[int, int],
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: str = "uint32",
    *,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int | None = None,
    maxFeatureChunk: int | None = None,
) -> zarr.Array:
    """Creates and returns a Zarr array with name 'counts'.

    Args:
        z (zarr.Group):
        assay_name (str):
        workspace (str | None):
        chunk_size (tuple[int, int]):
        n_cells (int):
        feat_ids (np.ndarray | list[str]):
        feat_names (np.ndarray | list[str]):
        dtype (str = 'uint32'):
        targetChunkBytes: Optional cloud feature-chunk byte target.
        minFeatureChunk: Optional lower clamp for feature-chunk width.
        maxFeatureChunk: Optional upper clamp for feature-chunk width.

    Returns:
        A Zarr array.
    """
    return _create_zarr_count_assay(
        z,
        assay_name,
        workspace,
        chunk_size,
        n_cells,
        feat_ids,
        feat_names,
        dtype,
        targetChunkBytes=targetChunkBytes,
        minFeatureChunk=minFeatureChunk,
        maxFeatureChunk=maxFeatureChunk,
    )


def finalize_writer_counts(
    store: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
) -> zarr.Array:
    """Finish durable assay counts and write feature-major ``countsT``."""
    return _finalize_counts(store, assay_name, workspace)


def load_count_store(
    z: zarr.Group, assay_name: str, workspace: str | None
) -> zarr.Array:
    """Return the counts Zarr array for an assay in the given workspace."""
    return _load_count_array(z, assay_name, workspace)


def create_cell_data(
    z: zarr.Group, workspace: str | None, ids: np.ndarray, names: np.ndarray
) -> zarr.Group:
    """Create the cellData group with ids, names, and default ``I`` filter column."""
    return _create_cell_data(z, workspace, ids, names)


def sparse_writer(
    store: zarr.Array,
    data_stream: Iterator[coo_matrix],
    n_cells: int,
    batch_size: int,
) -> int:
    """Write CSR batches into a Zarr count array row-wise.

    Returns:
        Number of rows written.
    """
    return accumulate_sparse_to_shards(store, data_stream, dtype=store.dtype)
