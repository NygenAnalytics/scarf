import math
import platform
import sys
from typing import Any

import numpy as np

from ..utils.arguments import integer_argument

# hnswlib 0.8 keeps one lock per label-operation slot, whatever the cell count.
_LABEL_OPERATION_LOCKS = 65_536
# Bytes of malloc's header and rounding at most, per upper-level link list.
_ALLOCATION_OVERHEAD_BYTES = 24
# The index's fixed objects, such as its distance space and search pools,
# and the Python wrapper: an allowance, measured below 16 KiB.
_ANN_INDEX_OBJECT_BYTES = 64 * 1024
# Bytes of the header that hnswlib writes before a saved index's cells.
_ANN_INDEX_HEADER_BYTES = 96


def _cxx_runtime_bytes() -> tuple[int, int]:
    """Return the bytes of a ``std::mutex`` and of a label-map entry here.

    ``std::mutex`` holds 40 bytes with glibc on x86-64 Linux, 48 on other
    Linux architectures such as arm64, 64 on macOS, and 80 with MSVC. An
    entry of the map from labels to cells holds a node and, while the map
    rehashes, up to three bucket slots: 56 bytes with libstdc++ and libc++,
    and 96 with MSVC, whose nodes link both ways and whose buckets hold two
    pointers.
    """
    if sys.platform == "win32":
        return 80, 96
    if sys.platform == "darwin":
        return 64, 56
    if platform.machine().lower() in {"x86_64", "amd64"}:
        return 40, 56
    return 48, 56


def _level0_bytes(dims: int, m: int) -> int:
    """Bytes of one cell in hnswlib's level-0 block.

    It holds ``2 * m`` uint32 links and their count, the cell's float32
    coordinates, and its 8-byte label.
    """
    return 8 * m + 4 + 4 * dims + 8


def _ann_index_shape(n_cells: int, dims: int, m: int) -> tuple[int, int, int]:
    return (
        integer_argument(n_cells, "n_cells", minimum=1),
        integer_argument(dims, "dims", minimum=1),
        integer_argument(m, "m", minimum=2),
    )


def ann_index_peak_bytes(
    n_cells: int,
    dims: int,
    m: int,
    *,
    nthreads: int = 1,
) -> int:
    """Estimate the most bytes an hnswlib index of ``n_cells`` cells holds.

    Args:
        n_cells: Number of cells the index holds.
        dims: Number of coordinate dimensions.
        m: The HNSW ``M``; at least two.
        nthreads: Threads that build or query the index.

    Returns:
        The estimated peak in bytes.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells, width, links = _ann_index_shape(n_cells, dims, m)
    threads = integer_argument(nthreads, "nthreads", minimum=1)
    mutex_bytes, label_entry_bytes = _cxx_runtime_bytes()
    upper = (4 * (links + 1) + _ALLOCATION_OVERHEAD_BYTES) / (links - 1)
    per_cell = (
        _level0_bytes(width, links)
        + mutex_bytes
        # The cell's level and the pointer to its upper-level links.
        + 4
        + 8
        + 2 * threads
        + label_entry_bytes
        + upper
    )
    return (
        math.ceil(cells * per_cell)
        + _LABEL_OPERATION_LOCKS * mutex_bytes
        + _ANN_INDEX_OBJECT_BYTES
    )


def ann_index_file_bytes(n_cells: int, dims: int, m: int) -> int:
    """Estimate the size of the file that hnswlib saves an index into.

    Args:
        n_cells: Number of cells the index holds.
        dims: Number of coordinate dimensions.
        m: The HNSW ``M``; at least two.

    Returns:
        The expected size in bytes.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If an argument is below its minimum.
    """
    cells, width, links = _ann_index_shape(n_cells, dims, m)
    upper = 4 * (links + 1) / (links - 1)
    return _ANN_INDEX_HEADER_BYTES + math.ceil(
        cells * (_level0_bytes(width, links) + 4 + upper)
    )


def ann_query_block_bytes(rows: int, k: int, *, ef: int, nthreads: int) -> int:
    """Estimate the bytes one block of a neighbor query holds beside the block.

    Each query thread also holds two candidate queues that grow with ``ef``.
    """
    cells = integer_argument(rows, "rows", minimum=1)
    neighbors = integer_argument(k, "k", minimum=1)
    depth = integer_argument(ef, "ef", minimum=1)
    threads = integer_argument(nthreads, "nthreads", minimum=1)
    # Two queues of 8-byte (distance, label) pairs per thread, with room for
    # candidates beyond the search depth and for their vectors to double.
    queues = threads * 64 * max(depth, neighbors)
    return cells * (27 * neighbors + 40) + queues


def instantiate_knn_index(
    space: str,
    dim: int,
    max_elements: int,
    ef_construction: int,
    M: int,
    random_seed: int,
    ef: int,
    nthreads: int,
) -> Any:
    """Create and configure an hnswlib KNN index."""
    import hnswlib

    ann_idx = hnswlib.Index(space=space, dim=dim)
    ann_idx.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )
    ann_idx.set_ef(ef)
    ann_idx.set_num_threads(nthreads)
    return ann_idx


def fix_knn_query(
    indices: np.ndarray,
    distances: np.ndarray,
    ref_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove self-neighbor entries from KNN query results."""
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    references = np.asarray(ref_idx)
    if (
        neighbor_indices.ndim != 2
        or neighbor_distances.shape != neighbor_indices.shape
        or references.shape != (neighbor_indices.shape[0],)
        or neighbor_indices.shape[1] < 2
    ):
        raise ValueError("KNN query arrays have incompatible shapes")
    matches = neighbor_indices == references[:, np.newaxis]
    has_self = matches.any(axis=1)
    self_positions = matches.argmax(axis=1)
    drop_positions = np.where(
        has_self,
        self_positions,
        neighbor_indices.shape[1] - 1,
    )
    columns = np.arange(neighbor_indices.shape[1])[np.newaxis, :]
    keep = columns != drop_positions[:, np.newaxis]
    output_shape = (neighbor_indices.shape[0], neighbor_indices.shape[1] - 1)
    fixed_indices = neighbor_indices[keep].reshape(output_shape)
    fixed_distances = neighbor_distances[keep].reshape(output_shape)
    missed_self_hits = int(np.count_nonzero(~has_self))
    return fixed_indices, fixed_distances, missed_self_hits
