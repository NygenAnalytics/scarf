"""Write small Scarf stores from sparse count matrices for tests."""

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.storage.io_policy import StorageIoPolicy
from scarf.storage.profiles import StorageProfile
from scarf.utils.logging import logger


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
    from scarf.storage.schema import (
        create_cell_data,
        create_zarr_count_assay,
        load_count_array,
        validate_assay_name,
    )
    from scarf.storage.sharding import write_dense_in_shard_rows
    from scarf.storage.budget import resolve_budget
    from scarf.storage.profiles import resolve_storage_profile
    from scarf.storage.stores import load_zarr

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
    from scarf.storage.identity import CountSummary, finalize_counts

    summary = CountSummary(store)
    write_dense_in_shard_rows(
        store,
        lambda s, e: sim_counts[s:e].toarray().astype(dtype),
        msg="Writing simulated doublets",
        resources=resources,
        io=io,
        residentBytes=sim_counts.data.nbytes
        + sim_counts.indices.nbytes
        + sim_counts.indptr.nbytes
        + summary.nbytes,
        producerBytes=min(n_sim, store.shards[0] if store.shards else store.chunks[0])
        * sim_counts.shape[1]
        * sim_counts.dtype.itemsize,
        countSummary=summary,
    )
    from scarf.assay.classification import (
        is_rna_assay_type,
        resolve_persisted_assay_type,
    )
    from scarf.storage.sharding import write_counts_t
    from scarf.storage.types import as_zarr_group

    type_name = resolve_persisted_assay_type(assay_name)
    raw_types = z.attrs.get("assayTypes", {})
    types = (
        {str(k): str(v) for k, v in raw_types.items()}
        if isinstance(raw_types, dict)
        else {}
    )
    types[assay_name] = type_name
    z.attrs["assayTypes"] = types
    finalize_counts(store, summary=summary)
    if is_rna_assay_type(type_name):
        group = as_zarr_group(z[assay_name], name=assay_name)
        write_counts_t(
            store,
            group,
            profile=resolved_profile,
            resources=resources,
            policy=policy,
            io=io,
        )
    logger.debug(f"Wrote {n_sim} simulated doublets to {zarr_loc}")
    return z
