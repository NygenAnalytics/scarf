from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import zarr

from .types import as_zarr_array, as_zarr_group, array_metadata_shards
from .arrays import create_numeric_array
from .layout import (
    DEFAULT_CLOUD_TARGET_CHUNK_BYTES,
    ZarrArraySpec,
    _CODEC_MAX_BYTES,
    _group_zarr_format,
    array_shard_rows,
    count_array_spec,
    get_compressors,
    iter_shard_row_slices,
)
from .profiles import StorageProfile, get_storage_profile


def write_dense_from_row_batches(
    dst: zarr.Array,
    batches: Iterator[np.ndarray],
    *,
    dtype: Any | None = None,
    msg: str | None = None,
) -> int:
    """Write row batches while flushing at destination shard boundaries."""
    from ..utils.progress import tqdmbar

    shard_rows = array_shard_rows(dst)
    pending: list[np.ndarray] = []
    pending_rows = 0
    out_pos = 0
    total_rows = 0

    def flush_partial(final: bool = False) -> None:
        nonlocal pending, pending_rows, out_pos, total_rows
        while pending_rows >= shard_rows or (final and pending_rows > 0):
            block = np.vstack(pending) if len(pending) > 1 else pending[0]
            take = block.shape[0] if final and pending_rows < shard_rows else shard_rows
            piece = block[:take]
            if dtype is not None:
                piece = piece.astype(dtype, copy=False)
            dst[out_pos : out_pos + piece.shape[0], :] = piece
            out_pos += piece.shape[0]
            total_rows += piece.shape[0]
            if block.shape[0] > take:
                pending = [block[take:]]
                pending_rows = block.shape[0] - take
            else:
                pending = []
                pending_rows = 0
            if not final and pending_rows < shard_rows:
                break

    for batch in tqdmbar(batches, desc=msg or "Writing Zarr array"):
        batch = np.asarray(batch)
        if batch.size == 0:
            continue
        pending.append(batch)
        pending_rows += batch.shape[0]
        flush_partial()
    flush_partial(final=True)
    return total_rows


def write_dense_in_shard_rows(
    dst: zarr.Array,
    produce: Callable[[int, int], np.ndarray],
    *,
    msg: str | None = None,
    shard_rows: int | None = None,
    also_write_to: zarr.Array | None = None,
) -> None:
    """Write a 2D array in row-shard bands."""
    from ..utils.progress import tqdmbar
    from .budget import shard_parallelism
    from .parallel import stream_shards

    n_rows = int(dst.shape[0])
    if n_rows == 0:
        return None
    if shard_rows is None:
        shard_rows = array_shard_rows(dst)
    slices = list(iter_shard_row_slices(n_rows, shard_rows))
    if msg is None:
        msg = "Writing Zarr array"

    plan = shard_parallelism(n_shards=len(slices))

    def produce_band(bounds: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        start, end = bounds
        return start, end, np.asarray(produce(start, end))

    produced = stream_shards(
        slices,
        produce_band,
        workers=plan.readAhead,
        within_block_threads=plan.withinBlockThreads,
        io_concurrency=plan.ioConcurrency,
    )
    for start, end, block in tqdmbar(produced, desc=msg, total=len(slices)):
        dst[start:end, :] = block
        if also_write_to is not None:
            also_write_to[start:end, :] = block
    return None


def accumulate_sparse_to_shards(
    dst: zarr.Array,
    data_stream: Iterator[Any],
    *,
    shard_rows: int | None = None,
    dtype: Any | None = None,
) -> int:
    """Buffer sparse COO batches and write one selection per row shard."""
    from scipy.sparse import coo_matrix

    if shard_rows is None:
        shard_rows = array_shard_rows(dst)
    shard_rows = max(1, int(shard_rows))
    if dtype is None:
        dtype = dst.dtype

    position = 0
    buffered_rows: list[np.ndarray] = []
    buffered_columns: list[np.ndarray] = []
    buffered_data: list[np.ndarray] = []
    next_flush = shard_rows

    def concat_or_empty(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts)

    def flush_through(end_row: int) -> None:
        nonlocal next_flush
        nonlocal buffered_rows, buffered_columns, buffered_data
        while next_flush <= end_row:
            band_start = next_flush - shard_rows
            row = concat_or_empty(buffered_rows)
            column = concat_or_empty(buffered_columns)
            data = concat_or_empty(buffered_data)
            if row.size:
                mask = (row >= band_start) & (row < next_flush)
                if mask.any():
                    dst.set_coordinate_selection(
                        (row[mask], column[mask]),
                        data[mask].astype(dtype, copy=False),
                    )
                keep = row >= next_flush
                if keep.any():
                    buffered_rows = [row[keep]]
                    buffered_columns = [column[keep]]
                    buffered_data = [data[keep]]
                else:
                    buffered_rows = []
                    buffered_columns = []
                    buffered_data = []
            next_flush += shard_rows

    for batch in data_stream:
        coo = batch if hasattr(batch, "row") else coo_matrix(batch)
        if coo.shape[0] == 0:
            continue
        if coo.nnz:
            buffered_rows.append(np.asarray(coo.row, dtype=np.int64) + position)
            buffered_columns.append(np.asarray(coo.col, dtype=np.int64))
            buffered_data.append(np.asarray(coo.data))
        position += coo.shape[0]
        flush_through(position)

    if buffered_rows:
        row = concat_or_empty(buffered_rows)
        column = concat_or_empty(buffered_columns)
        data = concat_or_empty(buffered_data)
        if row.size:
            dst.set_coordinate_selection(
                (row, column),
                data.astype(dtype, copy=False),
            )
    return position


def repack_to_sharded(
    srcArray: zarr.Array,
    dstGroup: zarr.Group,
    name: str,
    shards: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    profile: StorageProfile | None = None,
    overwrite: bool = True,
) -> zarr.Array:
    """Copy an array into a new sharded array."""
    chunks = chunks or tuple(srcArray.chunks)
    spec = ZarrArraySpec(
        shape=srcArray.shape,
        chunks=chunks,
        shards=shards,
        dtype=srcArray.dtype,
        compressors=get_compressors(profile or get_storage_profile()),
        overwrite=overwrite,
    )
    dstArray = create_numeric_array(dstGroup, name, spec)
    shardRows = int(shards[0])
    write_dense_in_shard_rows(
        dstArray,
        lambda start, end: np.asarray(srcArray[start:end, :]),
        msg="Repacking to sharded layout",
        shard_rows=shardRows,
    )
    return dstArray


def write_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
) -> zarr.Array | None:
    """Write feature-major ``countsT`` beside a finalized count matrix."""
    from ..utils.logging import logger
    from ..utils.progress import tqdmbar

    if _group_zarr_format(group) < 3:
        return None

    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    itemsize = max(1, int(np.dtype(counts.dtype).itemsize))
    source_row_chunk = max(1, int(counts.chunks[0]))
    feature_chunk = max(1, min(n_feats, int(counts.chunks[1])))

    bytes_per_row_band = feature_chunk * source_row_chunk * itemsize
    if bytes_per_row_band >= DEFAULT_CLOUD_TARGET_CHUNK_BYTES:
        cell_chunk = source_row_chunk
    else:
        cell_chunk = (
            DEFAULT_CLOUD_TARGET_CHUNK_BYTES // bytes_per_row_band
        ) * source_row_chunk
    cell_chunk = max(source_row_chunk, min(n_cells, cell_chunk))
    while (
        cell_chunk > source_row_chunk
        and feature_chunk * cell_chunk * itemsize > _CODEC_MAX_BYTES
    ):
        cell_chunk -= source_row_chunk
    cell_chunk = max(1, min(n_cells, cell_chunk))

    if "countsT" in group:
        del group["countsT"]
    counts_t = create_numeric_array(
        group,
        "countsT",
        ZarrArraySpec(
            shape=(n_feats, n_cells),
            chunks=(feature_chunk, cell_chunk),
            dtype=counts.dtype,
            shards=None,
            compressors=get_compressors(get_storage_profile()),
            fillValue=0,
            overwrite=True,
        ),
    )
    counts_t.attrs["complete"] = False
    logger.info(
        f"Writing countsT shape=({n_feats}, {n_cells}) "
        f"chunks=({feature_chunk}, {cell_chunk})"
    )

    feat_starts = list(range(0, n_feats, feature_chunk))
    cell_starts = list(range(0, n_cells, cell_chunk))
    total_tiles = max(1, len(feat_starts) * len(cell_starts))
    with tqdmbar(total=total_tiles, desc="Writing countsT") as progress:
        for feat_start in feat_starts:
            feat_end = min(feat_start + feature_chunk, n_feats)
            for cell_start in cell_starts:
                cell_end = min(cell_start + cell_chunk, n_cells)
                block = np.asarray(counts[cell_start:cell_end, feat_start:feat_end])
                counts_t[feat_start:feat_end, cell_start:cell_end] = block.T
                progress.update(1)

    counts_t.attrs["complete"] = True
    return counts_t


def finalize_sharded_counts(
    store: zarr.Group,
    assayName: str,
    workspace: str | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Repack assay counts to sharded layout when needed."""
    profile = profile or get_storage_profile()
    if workspace is None:
        countsPath = f"{assayName}/counts"
        assayGroup = as_zarr_group(store[assayName], name=assayName)
    else:
        countsPath = f"matrices/{assayName}/counts"
        assayGroup = as_zarr_group(store[f"matrices/{assayName}"], name=assayName)

    srcArray = as_zarr_array(store[countsPath], name=countsPath)
    if array_metadata_shards(srcArray) is not None:
        return srcArray
    if _group_zarr_format(store) == 2:
        return srcArray

    spec = count_array_spec(
        srcArray.shape[0],
        srcArray.shape[1],
        dtype=srcArray.dtype,
        profile=profile,
    )
    shards = spec.shards
    chunks = spec.chunks
    assert shards is not None and chunks is not None

    if shards == srcArray.shape:
        return srcArray
    tmpName = "counts__sharded_tmp"
    if tmpName in assayGroup:
        del assayGroup[tmpName]
    repack_to_sharded(
        srcArray,
        assayGroup,
        tmpName,
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup["counts"]
    repack_to_sharded(
        as_zarr_array(assayGroup[tmpName], name=tmpName),
        assayGroup,
        "counts",
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup[tmpName]

    if workspace is None:
        assayGroup = as_zarr_group(store[assayName], name=assayName)
    else:
        assayGroup = as_zarr_group(store[f"matrices/{assayName}"], name=assayName)
    assayGroup.attrs["scarf:zarr_spec"] = {
        "profile": profile,
        "chunks": list(chunks),
        "shards": list(shards),
        "zarr_format": 3,
    }
    return as_zarr_array(store[countsPath], name=countsPath)
