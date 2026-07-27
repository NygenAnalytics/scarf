"""Build persisted KNN query and graph arrays."""

from collections.abc import Generator

import numpy as np
import zarr

from ..storage.arrays import create_zarr_dataset
from ..storage.budget import READ_AHEAD
from ..storage.parallel import stream_shards
from ..utils.compute import controlled_compute
from ..utils.progress import tqdmbar
from .stream import AnnStream

__all__ = [
    "self_query_knn",
    "smoothen_dists",
]


def self_query_knn(
    ann_obj: AnnStream,
    store: zarr.Group,
    chunk_size: int,
    nthreads: int,
) -> float:
    """Query an ANN index and write its KNN arrays."""

    def get_transformed_data() -> Generator[np.ndarray, None, None]:
        msg = "Identifying neighbors"
        if ann_obj.embeddings is not None:
            bs = ann_obj.batchSize
            n_blocks = int(np.ceil(ann_obj.nCells / bs))
            for start in tqdmbar(
                range(0, ann_obj.nCells, bs),
                desc=msg,
                total=n_blocks,
            ):
                end = min(start + bs, ann_obj.nCells)
                yield ann_obj.embeddings[start:end]
            return
        if ann_obj.harmonizedData is None:
            source = ann_obj.data

            def transform(block: np.ndarray) -> np.ndarray:
                return ann_obj.reducer(controlled_compute(block, nthreads))
        else:
            source = ann_obj.harmonizedData

            def transform(block: np.ndarray) -> np.ndarray:
                return controlled_compute(block, nthreads)

        blocks = stream_shards(
            source.blocks,
            transform,
            workers=min(READ_AHEAD, max(1, nthreads)),
        )
        yield from tqdmbar(blocks, desc=msg, total=source.numblocks[0])

    from threadpoolctl import threadpool_limits

    from .query import self_query_blocks

    n_cells, n_neighbors = ann_obj.nCells, ann_obj.k
    z_knn = create_zarr_dataset(
        store,
        "indices",
        (chunk_size,),
        "u8",
        (n_cells, n_neighbors),
    )
    z_dist = create_zarr_dataset(
        store,
        "distances",
        (chunk_size,),
        "f8",
        (n_cells, n_neighbors),
    )
    missed_recall = 0
    with threadpool_limits(limits=nthreads):
        for start, end, indices, distances, missed in self_query_blocks(
            ann_obj,
            get_transformed_data(),
        ):
            z_knn[start:end, :] = indices
            z_dist[start:end, :] = distances
            missed_recall += missed
    recall = ann_obj.data.shape[0] - missed_recall
    return 100.0 * recall / ann_obj.data.shape[0]


def _patch_null_weights(
    zgw: zarr.Array,
    null_positions: list[int],
    fill_value: float,
    patch_chunk: int,
) -> None:
    """Patch zero edge weights without loading the full weights array."""
    if not null_positions:
        return
    null_positions_arr = np.asarray(null_positions, dtype=np.int64)
    n_weights = zgw.shape[0]
    for chunk_start in range(0, n_weights, patch_chunk):
        chunk_end = min(chunk_start + patch_chunk, n_weights)
        in_chunk = (null_positions_arr >= chunk_start) & (
            null_positions_arr < chunk_end
        )
        if not np.any(in_chunk):
            continue
        local_idx = null_positions_arr[in_chunk] - chunk_start
        block = np.asarray(zgw[chunk_start:chunk_end], dtype=np.float64)
        block[local_idx] = fill_value
        zgw[chunk_start:chunk_end] = block


def smoothen_dists(
    store: zarr.Group,
    z_idx: zarr.Array,
    z_dist: zarr.Array,
    lc: float,
    bw: float,
    chunk_size: int,
) -> None:
    """Smooth KNN distances and write graph edges and weights."""
    from .graph import smooth_knn_chunk

    n_cells, n_neighbors = z_idx.shape
    zge = create_zarr_dataset(
        store,
        "edges",
        (chunk_size * n_neighbors,),
        ("u8", "u8"),
        (n_cells * n_neighbors, 2),
    )
    zgw = create_zarr_dataset(
        store,
        "weights",
        (chunk_size * n_neighbors,),
        "f8",
        (n_cells * n_neighbors,),
    )
    last_row = 0
    value_count = 0
    null_positions: list[int] = []
    global_min = 1.0
    for start_row in tqdmbar(
        range(0, n_cells, chunk_size),
        desc="Smoothening KNN distances",
    ):
        end_row = min(start_row + chunk_size, n_cells)
        indices = np.asarray(z_idx[start_row:end_row, :])
        distances = np.asarray(z_dist[start_row:end_row, :])
        rows, cols, values = smooth_knn_chunk(
            indices,
            distances,
            local_connectivity=lc,
            bandwidth=bw,
        )
        rows = rows + last_row
        start = value_count
        end = value_count + len(rows)
        last_row = rows[-1] + 1
        value_count += len(rows)
        zge[start:end, 0] = rows
        zge[start:end, 1] = cols
        zgw[start:end] = values

        local_null = np.flatnonzero(values == 0)
        if local_null.size > 0:
            nonzero_values = values[values != 0]
            if nonzero_values.size > 0:
                global_min = min(global_min, float(nonzero_values.min()))
            null_positions.extend((start + local_null).tolist())

    _patch_null_weights(
        zgw,
        null_positions,
        global_min,
        chunk_size * n_neighbors,
    )
