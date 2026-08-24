"""Utilities for synthetic doublet detection."""

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile
from ..utils.logging import logger

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
    """Draw a per-cluster subsample of cell positions to seed doublet simulation."""
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
    """Generate index pairs into the candidate pool for simulated doublets."""
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
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> zarr.Group:
    """Materialise simulated doublet counts as a minimal Scarf Zarr hierarchy."""
    from ..storage.schema import (
        create_cell_data,
        create_zarr_count_assay,
        load_count_array,
        validate_assay_name,
    )
    from ..storage.sharding import write_dense_in_shard_rows
    from ..storage.budget import resolve_budget
    from ..storage.profiles import resolve_storage_profile
    from ..storage.stores import load_zarr

    validate_assay_name(assay_name)
    resources = resolve_budget(mem_budget, nthreads)
    resolved_profile = resolve_storage_profile(zarr_loc, profile)
    n_sim = sim_counts.shape[0]
    z = load_zarr(zarr_loc=zarr_loc, mode="w")
    ids = np.array([f"doublet_{i}" for i in range(n_sim)])
    create_cell_data(z, workspace=None, ids=ids, names=ids)
    create_zarr_count_assay(
        z=z,
        assay_name=assay_name,
        workspace=None,
        n_cells=n_sim,
        feat_ids=np.asarray(feat_ids),
        feat_names=np.asarray(feat_names),
        dtype=dtype,
        profile=resolved_profile,
        policy=policy,
    )
    store = load_count_array(z, assay_name, None)
    write_dense_in_shard_rows(
        store,
        lambda s, e: sim_counts[s:e].toarray().astype(dtype),
        msg="Writing simulated doublets",
        resources=resources,
        io=io,
    )
    from ..assay.classification import (
        is_rna_assay_type,
        resolve_persisted_assay_type,
    )
    from ..storage.sharding import finalize_rna_counts_t
    from ..storage.types import as_zarr_group

    type_name = resolve_persisted_assay_type(assay_name)
    raw_types = z.attrs.get("assayTypes", {})
    types = (
        {str(k): str(v) for k, v in raw_types.items()}
        if isinstance(raw_types, dict)
        else {}
    )
    types[assay_name] = type_name
    z.attrs["assayTypes"] = types
    if is_rna_assay_type(type_name):
        group = as_zarr_group(z[assay_name], name=assay_name)
        finalize_rna_counts_t(
            store,
            group,
            profile=resolved_profile,
            resources=resources,
            policy=policy,
            io=io,
        )
    logger.debug(f"Wrote {n_sim} simulated doublets to {zarr_loc}")
    return z
