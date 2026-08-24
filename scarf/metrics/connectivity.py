from collections.abc import Sequence

from numba import njit
import numpy as np
import pandas as pd
import zarr

from ..utils.progress import iter_progress
from ._types import ZarrArray

_CONNECTIVITY_BATCH_ROWS = 100_000


@njit(cache=True)
def _find_root(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def _union_same_label_edges(
    edges: np.ndarray,
    label_codes: np.ndarray,
    parent: np.ndarray,
    component_sizes: np.ndarray,
) -> None:
    for edge_index in range(edges.shape[0]):
        source = edges[edge_index, 0]
        target = edges[edge_index, 1]
        if label_codes[source] != label_codes[target]:
            continue

        source_root = _find_root(parent, source)
        target_root = _find_root(parent, target)
        if source_root == target_root:
            continue
        if component_sizes[source_root] < component_sizes[target_root]:
            source_root, target_root = target_root, source_root

        parent[target_root] = source_root
        component_sizes[source_root] += component_sizes[target_root]


@njit(cache=True)
def _compress_paths(parent: np.ndarray) -> None:
    for node in range(len(parent)):
        parent[node] = _find_root(parent, node)


def graph_connectivity(
    edges: np.ndarray | ZarrArray,
    labels: Sequence[object] | np.ndarray,
    batch_rows: int = _CONNECTIVITY_BATCH_ROWS,
) -> float:
    """Score label connectivity on a persisted, implicitly undirected graph.

    Each directed edge is treated as an undirected connection. The result is
    the mean, across labels, of the fraction of cells in the largest connected
    component for that label.

    This follows the original scIB symmetrized-graph definition. It does not
    match the directed strong-component calculation currently implemented by
    YosefLab ``scib-metrics``.

    References:
        Luecken et al. 2022 doi: 10.1038/s41592-021-01336-8
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Graph edges must have shape (n_edges, 2)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("Graph edges must contain integers")
    if batch_rows < 1:
        raise ValueError("batch_rows must be greater than zero")

    categorical = pd.Categorical(labels)
    n_cells = len(categorical)
    if n_cells == 0:
        raise ValueError("Graph connectivity requires at least one cell")
    if np.any(categorical.codes < 0):
        raise ValueError("Graph connectivity labels must not contain missing values")

    _, label_codes = np.unique(categorical.codes, return_inverse=True)
    label_codes = np.asarray(label_codes, dtype=np.int64)
    n_labels = int(label_codes.max()) + 1

    parent = np.arange(n_cells, dtype=np.int64)
    component_sizes = np.ones(n_cells, dtype=np.int64)
    chunk_rows = batch_rows
    if isinstance(edges, zarr.Array):
        chunk_rows = min(chunk_rows, int(edges.chunks[0]))

    n_edges = edges.shape[0]
    total = (n_edges + chunk_rows - 1) // chunk_rows
    for start in iter_progress(
        range(0, n_edges, chunk_rows),
        total=total,
        desc="Computing graph connectivity",
    ):
        end = min(start + chunk_rows, n_edges)
        edge_block = np.asarray(edges[start:end])
        if np.any(edge_block < 0) or np.any(edge_block >= n_cells):
            raise IndexError("Graph edge index is outside the label array")
        _union_same_label_edges(
            np.asarray(edge_block, dtype=np.int64),
            label_codes,
            parent,
            component_sizes,
        )

    _compress_paths(parent)
    roots = np.flatnonzero(parent == np.arange(n_cells))
    largest = np.zeros(n_labels, dtype=np.int64)
    np.maximum.at(largest, label_codes[roots], component_sizes[roots])
    label_sizes = np.bincount(label_codes, minlength=n_labels)
    return float(np.mean(largest / label_sizes))
