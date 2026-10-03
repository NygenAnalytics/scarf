"""Deterministic synthetic inputs shaped like the production funnel's data.

Sizes follow the recorded 1M-cell configuration: 1,000 highly variable
features, 21 principal components, and 11 neighbours. Builders are cached
per size so that benchmarks sharing an input build it once per session.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse

SEED = 4466
N_GENES = 2_000
N_HVGS = 1_000
N_DIMS = 21
N_NEIGHBORS = 11
N_GROUPS = 8
SIZE_FACTOR = 1_000.0


@lru_cache(maxsize=4)
def labelled_counts(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cells-by-genes uint16 counts and the program label of each cell.

    Cells draw Poisson counts from one of ``N_GROUPS`` expression programs
    with skewed gene means and about fourfold library-size spread, so most
    counts are zero, as in droplet RNA data.
    """
    rng = np.random.default_rng(SEED)
    scale = rng.gamma(0.25, 1.0, size=N_GENES)
    programs = scale * rng.lognormal(0.0, 1.0, size=(N_GROUPS, N_GENES))
    programs /= programs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, N_GROUPS, size=n_cells)
    library = rng.lognormal(np.log(1_500.0), 0.5, size=n_cells)
    counts = np.empty((n_cells, N_GENES), dtype=np.uint16)
    for start in range(0, n_cells, 4_096):
        stop = min(start + 4_096, n_cells)
        means = programs[labels[start:stop]] * library[start:stop, None]
        counts[start:stop] = np.minimum(rng.poisson(means), np.iinfo(np.uint16).max)
    return counts, labels


@lru_cache(maxsize=4)
def labelled_coordinates(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 coordinates in well-separated Gaussian clusters, and labels.

    The coordinates are shaped like PCA output, and the label of each cell
    is the cluster it was drawn from.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.normal(0.0, 4.0, size=(N_GROUPS, N_DIMS))
    labels = rng.integers(0, N_GROUPS, size=n_cells)
    values = centers[labels] + rng.normal(size=(n_cells, N_DIMS))
    return values.astype(np.float32), labels


@lru_cache(maxsize=4)
def neighbors(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``N_NEIGHBORS`` nearest other cells and their L2 distances."""
    from scarf.neighbors.index import fix_knn_query, instantiate_knn_index

    data, _labels = labelled_coordinates(n_cells)
    index = instantiate_knn_index(
        "l2",
        N_DIMS,
        n_cells,
        ef_construction=100,
        M=32,
        random_seed=SEED,
        ef=100,
        nthreads=1,
    )
    index.add_items(data, num_threads=1)
    indices, distances = index.knn_query(data, k=N_NEIGHBORS + 1, num_threads=1)
    indices, distances, _missed = fix_knn_query(
        indices, np.sqrt(distances), np.arange(n_cells)
    )
    return np.ascontiguousarray(indices, dtype=np.int64), np.ascontiguousarray(
        distances, dtype=np.float32
    )


@lru_cache(maxsize=4)
def connectivity_graph(n_cells: int) -> sparse.csr_matrix:
    """Return the directed fuzzy KNN graph that Scarf stores as a connectivity map."""
    from scarf.neighbors.graph import build_connectivity_arrays

    indices, distances = neighbors(n_cells)
    edges, weights = build_connectivity_arrays(
        indices, distances, local_connectivity=1.0, bandwidth=1.5
    )
    return sparse.csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)
    )


def initial_layout(n_cells: int) -> np.ndarray:
    """Return a deterministic two-dimensional UMAP starting layout."""
    data, _labels = labelled_coordinates(n_cells)
    return np.ascontiguousarray(data[:, :2] / np.abs(data[:, :2]).max() * 10.0)


def cluster_purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Return the fraction of cells whose cluster's majority label is their own."""
    matched = 0
    for cluster in np.unique(labels):
        matched += int(np.bincount(truth[labels == cluster]).max())
    return matched / truth.size
