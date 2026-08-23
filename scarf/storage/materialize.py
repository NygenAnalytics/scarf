import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import zarr

from ..utils.compute import controlled_compute
from .arrays import create_numeric_array, create_zarr_dataset
from .budget import ResourceBudget
from .layout import normed_array_spec
from .profiles import resolve_storage_profile
from .sharding import write_dense_from_row_batches, write_dense_in_shard_rows


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


def _normalize_count_block(
    block: np.ndarray,
    *,
    scaleFactor: float,
    logTransform: bool,
) -> np.ndarray:
    row_sum = block.sum(axis=1)
    row_sum[row_sum == 0] = 1
    normalized = np.empty(block.shape, dtype=np.float32)
    bytes_per_row = max(1, int(block.shape[1]) * np.dtype(np.float64).itemsize)
    rows_per_batch = max(1, (256 * 1024**2) // bytes_per_row)
    for start in range(0, int(block.shape[0]), rows_per_batch):
        end = min(start + rows_per_batch, int(block.shape[0]))
        work = scaleFactor * block[start:end] / row_sum[start:end, np.newaxis]
        if logTransform:
            np.log1p(work, out=work)
        normalized[start:end] = work
    return normalized


def _counts_t_renormalized_batches(
    assay: Any,
    cellIdx: np.ndarray,
    featIdx: np.ndarray,
    *,
    scaleFactor: float,
    logTransform: bool,
) -> Iterator[np.ndarray]:
    from .feature_stream import map_feature_cell_bands, selected_feature_values

    counts_t = assay.rawDataT
    if counts_t is None:
        raise ValueError("Feature-major normalization requires countsT")
    selected_cells = np.asarray(cellIdx, dtype=np.int64)
    selected_features = np.asarray(featIdx, dtype=np.int64)
    if selected_cells.size > 1 and np.any(np.diff(selected_cells) <= 0):
        raise ValueError("Feature-major normalization requires sorted unique cells")

    n_features = int(selected_features.shape[0])
    feature_destinations = np.full(int(counts_t.shape[0]), -1, dtype=np.int64)
    feature_destinations[selected_features] = np.arange(n_features, dtype=np.int64)
    cell_chunk = max(1, int(counts_t.chunks[1]))
    raw_band_bytes = (
        cell_chunk * n_features * max(1, int(np.dtype(counts_t.dtype).itemsize))
    )
    normalized_band_bytes = cell_chunk * n_features * np.dtype(np.float32).itemsize
    scratch_bytes = (
        2 * raw_band_bytes
        + normalized_band_bytes
        + min(256 * 1024**2, 2 * raw_band_bytes)
    )

    lock = threading.Lock()
    buffers: dict[int, np.ndarray] = {}
    completed_groups: dict[int, int] = {}
    metrics: dict[str, Any] = {}

    def fill_band(band: Any) -> tuple[int, np.ndarray] | None:
        local_dest = feature_destinations[band.featStart : band.featEnd]
        keep = local_dest >= 0
        if not np.any(keep):
            raise RuntimeError(
                "Planned countsT group did not contain selected features"
            )
        row_destinations = np.asarray(band.selectedDestinations, dtype=np.int64)
        row_start = int(row_destinations[0])
        expected_rows = np.arange(
            row_start,
            row_start + int(row_destinations.shape[0]),
            dtype=np.int64,
        )
        if not np.array_equal(row_destinations, expected_rows):
            raise ValueError(
                "Feature-major normalization requires contiguous selected-cell bands"
            )
        with lock:
            raw = buffers.get(band.cellStart)
            if raw is None:
                raw = np.empty(
                    (int(row_destinations.shape[0]), n_features),
                    dtype=counts_t.dtype,
                )
                buffers[band.cellStart] = raw
                completed_groups[band.cellStart] = 0
        selected = selected_feature_values(band.values, keep)
        destinations = local_dest[keep]
        raw[:, destinations] = selected[:, band.selectedLocal].T
        with lock:
            completed_groups[band.cellStart] += 1
            group_count = int(metrics["featureGroupCount"])
            if completed_groups[band.cellStart] != group_count:
                return None
            del completed_groups[band.cellStart]
            return row_start, buffers.pop(band.cellStart)

    pending: dict[int, np.ndarray] = {}
    next_row = 0
    for item in map_feature_cell_bands(
        counts_t,
        fill_band,
        cell_idx=selected_cells,
        feat_idx=selected_features,
        resources=assay.resources,
        io=getattr(assay, "storageIo", None),
        metrics=metrics,
        scratchBytes=scratch_bytes,
        orderedCompute=False,
        cellMajorOrder=True,
    ):
        if item is None:
            continue
        row_start, raw = item
        pending[row_start] = _normalize_count_block(
            raw,
            scaleFactor=scaleFactor,
            logTransform=logTransform,
        )
        while next_row in pending:
            normalized = pending.pop(next_row)
            next_row += int(normalized.shape[0])
            yield normalized
    if (
        buffers
        or completed_groups
        or pending
        or next_row != int(selected_cells.shape[0])
    ):
        raise RuntimeError(
            "Feature-major normalization did not cover every selected cell"
        )


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
    if scale_factor is None:
        raise ValueError("Library-size normalization requires a size factor")

    if getattr(assay, "rawDataT", None) is not None and mirror is None:
        summary: tuple[np.ndarray, np.ndarray] | None = None

        def normalized_batches() -> Iterator[np.ndarray]:
            nonlocal summary
            for block in _counts_t_renormalized_batches(
                assay,
                cell_idx,
                feat_idx,
                scaleFactor=float(scale_factor),
                logTransform=log_transform,
            ):
                if stats_group is not None:
                    current = _feature_summary(block)
                    summary = (
                        current
                        if summary is None
                        else _merge_feature_summaries(summary, current)
                    )
                yield block

        write_dense_from_row_batches(
            output,
            normalized_batches(),
            resources=assay.resources,
            io=getattr(assay, "storageIo", None),
            msg=msg,
        )
        _write_feature_summaries(stats_group, summary)
        return

    def normalize_block(block: Any) -> np.ndarray:
        return _normalize_count_block(
            np.asarray(block),
            scaleFactor=float(scale_factor),
            logTransform=log_transform,
        )

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
        io=getattr(assay, "storageIo", None),
    )
    _write_feature_summaries(stats_group, summary)


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
        io=getattr(data, "_io", None),
    )
    _write_feature_summaries(stats_group, summary)
