from typing import Any

import numpy as np
import zarr

from ..utils.compute import controlled_compute
from ..utils.moments import ColumnMoments, column_moments
from .arrays import create_numeric_array, create_zarr_dataset
from .budget import ResourceBudget, resolve_budget
from .layout import array_shard_rows, normed_array_spec
from .profiles import resolve_storage_profile
from .sharding import write_dense_in_shard_rows


def feature_summary_bytes(n_features: int) -> int:
    """Return the bytes that summarizing normalized features holds at most."""
    return 7 * int(n_features) * np.dtype(np.float64).itemsize


def _feature_summary(block: np.ndarray) -> ColumnMoments:
    return column_moments(block)


def _merge_feature_summaries(
    accumulated: ColumnMoments,
    current: ColumnMoments,
) -> ColumnMoments:
    return accumulated.merge(current)


def _write_feature_summaries(
    group: zarr.Group | None,
    summary: ColumnMoments | None,
) -> None:
    """Store the sum and ``m2`` of each normalized feature beside its data.

    ``feature_m2`` holds each feature's sum of squared deviations from its
    mean over the normalized cells, so feature scaling reads the variance
    ``feature_m2 / n_cells`` without the cancellation of squared sums.
    """
    if group is None or summary is None:
        return
    for name, values in (
        ("feature_sum", summary.total),
        ("feature_m2", summary.m2),
    ):
        output = create_zarr_dataset(
            group,
            name,
            (100_000,),
            np.float64,
            values.shape,
        )
        output[:] = values


def chunked_to_zarr(
    data: Any,
    root: zarr.Group,
    loc: str,
    nthreads: int,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
    resources: ResourceBudget | None = None,
    stats_group: zarr.Group | None = None,
    *,
    requireFinite: bool = False,
    operation: str | None = None,
) -> None:
    """Write a chunked matrix as a float32 array of normalized-data layout.

    Args:
        data: Chunked matrix to write.
        root: Group in which the array is created.
        loc: Path of the new array in ``root``.
        nthreads: Maximum number of threads for computing and writing.
        msg: Progress message; by default it names ``loc``.
        mirror: Second array that receives the same values.
        resources: Memory and worker budget for the write.
        stats_group: Group that receives the sum and ``m2`` of each column.
        requireFinite: Refuse values that are NaN or infinite as float32.
        operation: Producer of the values, required with ``requireFinite``.

    Raises:
        ValueError: If ``requireFinite`` is set and a value is not finite.
    """
    if msg is None:
        msg = f"Writing data to {loc}"
    spec = normed_array_spec(
        data.shape[0],
        data.shape[1],
        profile=resolve_storage_profile(root.store),
    )
    output = create_numeric_array(root, loc, spec)
    budget = resources or resolve_budget(workers=nthreads)
    budget = ResourceBudget(budget.memoryBytes, min(max(1, nthreads), budget.workers))
    band = data[: array_shard_rows(output), :]._with_block_size(
        array_shard_rows(output)
    )
    producer_bytes = band._block_task_bytes()
    summary = write_dense_in_shard_rows(
        output,
        lambda start, end: controlled_compute(
            data[start:end, :],
            nthreads,
        ).astype(np.float32, copy=False),
        msg=msg,
        also_write_to=mirror,
        resources=budget,
        residentBytes=data._resident_bytes(),
        producerBytes=producer_bytes,
        resultBytes=(
            feature_summary_bytes(data.shape[1]) if stats_group is not None else 0
        ),
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
        io=getattr(data, "_io", None),
        requireFinite=requireFinite,
        operation=operation,
    )
    _write_feature_summaries(stats_group, summary)
