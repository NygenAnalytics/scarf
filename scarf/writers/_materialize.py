from typing import Any

import numpy as np
import zarr

from ..storage.budget import ResourceBudget
from ..storage.materialize import (
    dask_to_zarr as _dask_to_zarr,
    write_renorm_subset_to_zarr as _write_renorm_subset_to_zarr,
)


def write_renorm_subset_to_zarr(
    assay: Any,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    z: zarr.Group,
    loc: str,
    nthreads: int,
    log_transform: bool = False,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
) -> None:
    """Write library-size normalized subset data in a single scattered read pass.

    For HVG subsets the per-cell scale factor is the row sum within each block,
    so a separate ``counts.sum(axis=1)`` pass is not needed before writing.

    Args:
        assay: Source assay providing counts and library sizes.
        cell_idx: Cell indices to write.
        feat_idx: Feature indices to write.
        z: Destination Zarr group.
        loc: Array path within the group.
        nthreads: Threads for block compute.
        log_transform: If True, write log1p normalized values.
        msg: Optional progress message.
        mirror: Optional second array of the same shape written in the same pass.
    """
    return _write_renorm_subset_to_zarr(
        assay,
        cell_idx,
        feat_idx,
        z,
        loc,
        nthreads,
        log_transform,
        msg=msg,
        mirror=mirror,
    )


def dask_to_zarr(
    df: Any,
    z: zarr.Group,
    loc: str,
    nthreads: int,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
    resources: ResourceBudget | None = None,
) -> None:
    """Creates a Zarr hierarchy from a chunked array.

    Args:
        df: ChunkedArray to materialize and write.
        z: Root Zarr group.
        loc: Array path within the group.
        nthreads: Threads for block compute.
        msg: Progress bar message (default: ``Writing data to {loc}``).
        mirror: Optional second array of the same shape to write each band into
            during the same pass (local staging cache).
        resources: Optional memory and worker budget for the write.
    """
    return _dask_to_zarr(
        df,
        z,
        loc,
        nthreads,
        msg=msg,
        mirror=mirror,
        resources=resources,
    )
