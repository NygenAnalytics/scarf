from typing import Any

import numpy as np
import zarr

from ..utils.compute import controlled_compute
from .arrays import create_numeric_array, create_zarr_dataset
from .budget import ResourceBudget
from .layout import normed_array_spec
from .profiles import resolve_storage_profile
from .sharding import write_dense_in_shard_rows


def _feature_summary(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(np.sum(block, axis=0, dtype=np.float64)),
        np.asarray(
            np.einsum(
                "ij,ij->j",
                block,
                block,
                dtype=np.float64,
                optimize=True,
            )
        ),
    )


def _merge_feature_summaries(
    accumulated: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    accumulated[0][...] += current[0]
    accumulated[1][...] += current[1]
    return accumulated


def _write_feature_summaries(
    group: zarr.Group | None,
    summary: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if group is None or summary is None:
        return
    for name, values in zip(
        ("feature_sum", "feature_squared_sum"),
        summary,
        strict=True,
    ):
        output = create_zarr_dataset(
            group,
            name,
            (100_000,),
            np.float64,
            values.shape,
        )
        output[:] = values


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
    stats_group: zarr.Group | None = None,
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

    summary = write_dense_in_shard_rows(
        output,
        lambda start, end: normalize_block(
            controlled_compute(counts[start:end, :], nthreads)
        ),
        msg=msg,
        also_write_to=mirror,
        resources=assay.resources,
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
    )
    _write_feature_summaries(stats_group, summary)


def dask_to_zarr(
    data: Any,
    root: zarr.Group,
    loc: str,
    nthreads: int,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
    resources: ResourceBudget | None = None,
    stats_group: zarr.Group | None = None,
) -> None:
    if msg is None:
        msg = f"Writing data to {loc}"
    spec = normed_array_spec(
        data.shape[0],
        data.shape[1],
        profile=resolve_storage_profile(root.store),
    )
    output = create_numeric_array(root, loc, spec)
    summary = write_dense_in_shard_rows(
        output,
        lambda start, end: controlled_compute(
            data[start:end, :],
            nthreads,
        ).astype(np.float32, copy=False),
        msg=msg,
        also_write_to=mirror,
        resources=resources,
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
    )
    _write_feature_summaries(stats_group, summary)
