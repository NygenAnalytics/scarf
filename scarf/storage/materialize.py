from typing import Any

import numpy as np
import zarr

from ..utils.compute import controlled_compute
from ..utils.arrays import sum_and_squared_sum
from .arrays import create_numeric_array, create_zarr_dataset
from .budget import ResourceBudget, resolve_budget
from .layout import array_shard_rows, normed_array_spec
from .profiles import resolve_storage_profile
from .sharding import write_dense_in_shard_rows


def _feature_summary(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sum_and_squared_sum(block)


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


def chunked_to_zarr(
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
        resultBytes=2 * data.shape[1] * 8 if stats_group is not None else 0,
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
        io=getattr(data, "_io", None),
    )
    _write_feature_summaries(stats_group, summary)
