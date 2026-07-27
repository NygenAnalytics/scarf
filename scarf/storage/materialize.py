from typing import Any

import numpy as np
import zarr

from ..utils.compute import controlled_compute
from .arrays import create_numeric_array
from .budget import ResourceBudget
from .layout import normed_array_spec
from .profiles import resolve_storage_profile
from .sharding import write_dense_in_shard_rows


def write_renorm_subset_to_zarr(
    assay: Any,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    root: zarr.Group,
    loc: str,
    nthreads: int,
    log_transform: bool = False,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
) -> None:
    counts = assay.rawData[:, feat_idx][cell_idx, :]
    if msg is None:
        msg = f"Writing data to {loc}"
    spec = normed_array_spec(
        counts.shape[0],
        counts.shape[1],
        profile=resolve_storage_profile(root.store),
    )
    output = create_numeric_array(root, loc, spec)
    scale_factor = assay.sf

    def normalize_block(block: Any) -> np.ndarray:
        block = np.asarray(block)
        row_sum = block.sum(axis=1)
        row_sum[row_sum == 0] = 1
        normalized = scale_factor * block / row_sum[:, np.newaxis]
        if log_transform:
            normalized = np.log1p(normalized)
        return np.asarray(normalized, dtype=np.float32)

    write_dense_in_shard_rows(
        output,
        lambda start, end: normalize_block(
            controlled_compute(counts[start:end, :], nthreads)
        ),
        msg=msg,
        also_write_to=mirror,
        resources=assay.resources,
    )


def dask_to_zarr(
    data: Any,
    root: zarr.Group,
    loc: str,
    nthreads: int,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
    resources: ResourceBudget | None = None,
) -> None:
    if msg is None:
        msg = f"Writing data to {loc}"
    spec = normed_array_spec(
        data.shape[0],
        data.shape[1],
        profile=resolve_storage_profile(root.store),
    )
    output = create_numeric_array(root, loc, spec)
    write_dense_in_shard_rows(
        output,
        lambda start, end: controlled_compute(
            data[start:end, :],
            nthreads,
        ).astype(np.float32, copy=False),
        msg=msg,
        also_write_to=mirror,
        resources=resources,
    )
