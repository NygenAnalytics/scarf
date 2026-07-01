"""Utilities for synthetic doublet detection.

This implements a Scrublet/DoubletFinder style strategy adapted to Scarf's
out-of-core graph and projection framework. Synthetic doublets are simulated by
summing pairs of observed transcriptomes, projected onto the existing reference
KNN graph and scored by how often each reference cell is found among the
nearest neighbours of the simulated doublets. The scores are subsequently
diffused over the graph using the same operator that powers `get_imputed`.
"""

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .utils import logger

__all__ = [
    "sample_cluster_pool",
    "simulate_doublet_pairs",
    "write_doublet_target_zarr",
]


def sample_cluster_pool(
    clusters: NDArray,
    fraction: float,
    max_per_cluster: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw a per-cluster subsample of cell positions to seed doublet simulation.

    For each cluster a fraction of its cells is sampled, capped at
    `max_per_cluster`. The returned positions index into the array of clusters
    that was passed in (i.e. positions among the active reference cells).

    Args:
        clusters: Cluster label for each active reference cell.
        fraction: Fraction of each cluster to sample.
        max_per_cluster: Hard cap on the number of cells sampled per cluster.
        rng: Random number generator.

    Returns:
        Sorted array of sampled cell positions.
    """
    pool = []
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        n = min(int(np.ceil(len(idx) * fraction)), max_per_cluster, len(idx))
        if n <= 0:
            continue
        pool.append(rng.choice(idx, size=n, replace=False))
    if len(pool) == 0:
        raise ValueError("ERROR: No cells could be sampled to simulate doublets")
    return np.sort(np.concatenate(pool))


def simulate_doublet_pairs(
    pool_clusters: NDArray,
    n_sim: int,
    heterotypic_fraction: float,
    rng: np.random.Generator,
    max_tries: int = 20,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generate index pairs into the candidate pool for simulated doublets.

    A configurable fraction of the pairs are biased to be heterotypic (the two
    cells coming from different clusters), because homotypic doublets are
    largely indistinguishable from singlets and carry little detection signal.

    Args:
        pool_clusters: Cluster label for each cell in the candidate pool.
        n_sim: Number of doublet pairs to generate.
        heterotypic_fraction: Fraction of pairs forced to be cross-cluster.
        rng: Random number generator.
        max_tries: Maximum resampling rounds used to satisfy the heterotypic
            constraint before falling back to whatever was drawn.

    Returns:
        A tuple of two arrays holding the left and right pool indices.
    """
    pool_size = len(pool_clusters)
    left = rng.integers(0, pool_size, size=n_sim)
    right = rng.integers(0, pool_size, size=n_sim)
    if heterotypic_fraction > 0 and len(np.unique(pool_clusters)) > 1:
        want_hetero = rng.random(n_sim) < heterotypic_fraction
        for _ in range(max_tries):
            clash = want_hetero & (pool_clusters[left] == pool_clusters[right])
            if not clash.any():
                break
            right[clash] = rng.integers(0, pool_size, size=int(clash.sum()))
    return left, right


def write_doublet_target_zarr(
    zarr_loc: str,
    assay_name: str,
    sim_counts: csr_matrix,
    feat_ids: NDArray,
    feat_names: NDArray,
    dtype: str = "uint32",
    batch_size: int = 1000,
) -> zarr.Group:
    """Materialise simulated doublet counts as a minimal Scarf Zarr hierarchy.

    The resulting store mirrors the feature universe of the reference assay so
    that `run_mapping` can align features by id without rebuilding the graph.

    Args:
        zarr_loc: Destination path (or store) for the temporary Zarr hierarchy.
        assay_name: Name to give the simulated assay (matched to the reference).
        sim_counts: Sparse matrix of simulated doublet counts (doublets x genes).
        feat_ids: Feature ids in the same order as the reference raw matrix.
        feat_names: Feature names in the same order as the reference raw matrix.
        dtype: Storage dtype for the count matrix.
        batch_size: Number of rows written per batch.

    Returns:
        The root Zarr group of the written store.
    """
    from .storage.zarr_store import write_dense_in_shard_rows
    from .utils import load_zarr
    from .writers import (
        create_cell_data,
        create_zarr_count_assay,
        finalize_writer_counts,
        load_count_store,
    )

    n_sim = sim_counts.shape[0]
    z = load_zarr(zarr_loc=zarr_loc, mode="w")
    ids = np.array([f"doublet_{i}" for i in range(n_sim)])
    create_cell_data(z, workspace=None, ids=ids, names=ids)
    create_zarr_count_assay(
        z=z,
        assay_name=assay_name,
        workspace=None,
        chunk_size=(batch_size, 1000),
        n_cells=n_sim,
        feat_ids=np.asarray(feat_ids),
        feat_names=np.asarray(feat_names),
        dtype=dtype,
    )
    store = load_count_store(z, assay_name, None)
    write_dense_in_shard_rows(
        store,
        lambda s, e: sim_counts[s:e].toarray().astype(dtype),
        msg="Writing simulated doublets",
    )
    finalize_writer_counts(z, assay_name, None)
    logger.debug(f"Wrote {n_sim} simulated doublets to {zarr_loc}")
    return z
