"""Utilities for synthetic doublet detection."""

from collections.abc import Iterator

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, vstack
from threadpoolctl import threadpool_limits

from ..assay import RNAassay
from ..assay.persistence import _read_block
from ..mapping.artifact import _load_reference_neighbor_query, _reference_available_k
from ..mapping.confidence import mapping_score_weights
from ..mapping.features import normalize_reference_counts
from ..mapping.reference import MappingReference
from ..mapping.symphony import project_pca, zero_norm_rows
from ..matrix import ChunkedArray
from ..neighbors.diffusion import diffusion_operator
from ..storage.ann_index import ANN_INDEX_ARRAY, ANN_INDEX_CHUNK_BYTES
from ..storage.artifacts import artifact_group
from ..storage.budget import ResourceBudget, admit_stream
from ..storage.geometry import array_geometry
from ..storage.parallel import stream_shards
from ..storage.partition import affordable_width, row_band
from ..storage.types import as_zarr_array

from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile
from ..utils.logging import logger

__all__ = [
    "sample_cluster_pool",
    "simulate_doublet_pairs",
    "sum_doublet_pairs",
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


def sum_doublet_pairs(
    pool_counts: csr_matrix,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
) -> csr_matrix:
    """Add sampled count rows without overflowing integer counts."""
    dtype = pool_counts.dtype
    if dtype.kind == "b":
        dtype = np.dtype("uint8")
    elif dtype.kind in "iu" and dtype.itemsize < 8:
        dtype = np.dtype(f"{dtype.kind}{dtype.itemsize * 2}")
    first = pool_counts[left].astype(dtype, copy=False)
    second = pool_counts[right].astype(dtype, copy=False)
    if dtype.kind in "iu" and pool_counts.dtype.itemsize == 8 and first.nnz:
        entries = first.tocoo(copy=False)
        other_values = np.asarray(second[entries.row, entries.col]).ravel()
        limits = np.iinfo(dtype)
        positive = other_values > 0
        overflow = np.any(entries.data[positive] > limits.max - other_values[positive])
        if dtype.kind == "i":
            negative = other_values < 0
            overflow |= np.any(
                entries.data[negative] < limits.min - other_values[negative]
            )
        if overflow:
            raise OverflowError(f"Synthetic doublet counts exceed the range of {dtype}")
    return (first + second).tocsr()


def _csr_bytes(values: csr_matrix) -> int:
    return int(values.data.nbytes + values.indices.nbytes + values.indptr.nbytes)


def _doublet_batch_rows(
    resources: ResourceBudget,
    preferred: int,
    row_bytes: int,
    resident_bytes: int,
    decode_bytes: int = 0,
) -> int:
    rows = affordable_width(
        lambda n: (
            resident_bytes + n * row_bytes + decode_bytes <= resources.memoryBytes
        ),
        preferred,
    )
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=max(1, rows) * row_bytes,
        decodeBytes=decode_bytes,
        residentBytes=resident_bytes,
        requested=1,
    )
    return rows


def _load_parent_counts(
    raw: ChunkedArray,
    cell_indices: np.ndarray,
    resources: ResourceBudget,
    resident_bytes: int,
) -> csr_matrix:
    backing = raw._backing
    geometry = array_geometry(backing)
    preferred = row_band(geometry, unit="chunk", fallback=1_000)
    decode_bytes = 0 if geometry is None else geometry.nominalChunkBytes()
    columns = np.arange(raw.shape[1], dtype=np.int64)
    parts: list[csr_matrix] = []
    retained = 0
    # Dense read, sparse conversion scratch, and the eventual concatenation copy.
    row_bytes = len(columns) * (4 * (raw.dtype.itemsize + 8)) + 32

    def boundaries() -> Iterator[tuple[int, int]]:
        start = 0
        while start < len(cell_indices):
            rows = _doublet_batch_rows(
                resources,
                min(preferred, len(cell_indices) - start),
                row_bytes,
                resident_bytes + columns.nbytes + 2 * retained,
                decode_bytes,
            )
            yield start, start + rows
            start += rows

    def read(interval: tuple[int, int]) -> csr_matrix:
        start, end = interval
        rows = cell_indices[start:end]
        values = (
            backing[np.ix_(rows, columns)]
            if isinstance(backing, np.ndarray)
            else _read_block(backing, rows, columns)
        )
        return csr_matrix(values)

    for part in stream_shards(
        boundaries(),
        read,
        workers=1,
        io_concurrency=1,
        msg="Reading doublet parents",
    ):
        parts.append(part)
        # Concatenation may promote sparse indices to int64.
        retained += part.nnz * (raw.dtype.itemsize + 8) + 8 * (part.shape[0] + 1)
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=max(1, retained),
        residentBytes=resident_bytes + retained,
        requested=1,
    )
    return vstack(parts, format="csr")


def _pair_row_bytes(pool: csr_matrix) -> int:
    largest = int(np.diff(pool.indptr).max(initial=0))
    # Row gathers, widening, addition, and checked 64-bit overflow scratch.
    return int(6 * (2 * largest * (max(8, pool.dtype.itemsize) + 8) + 16))


def _doublet_statistics(
    pool: csr_matrix,
    left: np.ndarray,
    right: np.ndarray,
    *,
    resources: ResourceBudget,
    preferred_rows: int,
    resident_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_rows = _doublet_batch_rows(
        resources,
        min(preferred_rows, len(left)),
        _pair_row_bytes(pool),
        resident_bytes + _csr_bytes(pool) + 24 * len(left),
    )
    totals = np.empty(len(left), dtype=np.float64)
    n_features = np.empty(len(left), dtype=np.int64)
    for start in range(0, len(left), batch_rows):
        stop = min(start + batch_rows, len(left))
        simulated = sum_doublet_pairs(pool, left[start:stop], right[start:stop])
        totals[start:stop] = np.asarray(simulated.sum(axis=1, dtype=np.float64)).ravel()
        n_features[start:stop] = np.asarray((simulated > 0).sum(axis=1)).ravel()
        del simulated
    keep = n_features > 10
    if np.median(n_features, overwrite_input=True) < 10:
        keep[:] = True
    if not keep.any():
        raise ValueError("No synthetic doublets pass the minimum-feature filter")
    return totals, keep


def score_synthetic_doublets(
    assay: RNAassay,
    reference: MappingReference,
    active_indices: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    *,
    cluster_sample_fraction: float,
    max_cells_per_cluster: int,
    simulation_ratio: float,
    heterotypic_fraction: float,
    save_k: int,
    random_seed: int,
    resources: ResourceBudget,
    reserved_resident_bytes: int = 0,
) -> np.ndarray:
    if isinstance(save_k, bool | np.bool_) or not isinstance(save_k, int | np.integer):
        raise TypeError("save_k must be a positive integer")
    if save_k < 1:
        raise ValueError("save_k must be a positive integer")
    available_k = _reference_available_k(reference)
    if save_k > available_k:
        logger.warning(f"`save_k` was decreased to {available_k}")
    n_k = min(int(save_k), available_k)
    model = reference.model
    resident = reserved_resident_bytes + sum(
        values.nbytes
        for values in (
            active_indices,
            labels,
            feature_indices,
            reference.feature_ids,
            reference.reference_distance_quantiles,
            reference.reference_distance_values,
            model.loadings,
            model.feature_means,
            model.feature_scales,
            model.center,
        )
    )
    n_sim = max(1, int(round(simulation_ratio * len(active_indices))))
    # Pair generation temporarily gathers both label vectors and rejection masks.
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=n_sim * (48 + 2 * labels.dtype.itemsize),
        residentBytes=resident + 64 * len(active_indices),
        requested=1,
    )
    rng = np.random.default_rng(random_seed)
    pool_positions = sample_cluster_pool(
        labels,
        cluster_sample_fraction,
        max_cells_per_cluster,
        rng,
    )
    left, right = simulate_doublet_pairs(
        labels[pool_positions],
        n_sim,
        heterotypic_fraction,
        rng,
    )
    resident += left.nbytes + right.nbytes + 16 * len(pool_positions)
    pool = _load_parent_counts(
        assay.rawData,
        active_indices[pool_positions],
        resources,
        resident,
    )
    preferred = row_band(
        array_geometry(assay.rawData._backing),
        unit="chunk",
        fallback=1_000,
    )
    totals, keep = _doublet_statistics(
        pool,
        left,
        right,
        resources=resources,
        preferred_rows=preferred,
        resident_bytes=resident,
    )
    resident += totals.nbytes + keep.nbytes
    subset_bytes = min(pool.nnz, pool.shape[0] * len(feature_indices)) * (
        pool.dtype.itemsize + 8
    ) + 8 * (pool.shape[0] + 1)
    admit_stream(
        resources,
        nBlocks=1,
        blockBytes=max(
            1, 2 * subset_bytes + 8 * (pool.shape[1] + len(feature_indices))
        ),
        residentBytes=resident + _csr_bytes(pool),
        requested=1,
    )
    selected_pool = pool[:, feature_indices].tocsr()
    del pool
    resident += _csr_bytes(selected_pool) + 8 * len(active_indices)

    ann = as_zarr_array(
        artifact_group(reference.datastore.zw, reference.ann_index)[ANN_INDEX_ARRAY],
        name=ANN_INDEX_ARRAY,
    )
    # Serialized index plus native structure/allocator and per-worker visited arrays.
    resident += (
        2 * int(ann.nbytes)
        + ANN_INDEX_CHUNK_BYTES
        + reference.selected_cell_count * (64 + 2 * resources.workers)
    )
    row_bytes = (
        _pair_row_bytes(selected_pool)
        + model.n_features * (max(8, selected_pool.dtype.itemsize) + 3 * 8)
        + model.n_dims * 32
        + n_k * 64
        + 64
    )
    batch_rows = _doublet_batch_rows(
        resources,
        min(preferred, n_sim),
        row_bytes,
        resident,
    )
    query = _load_reference_neighbor_query(
        reference,
        save_k=n_k,
        workers=resources.workers,
    )
    scores = np.zeros(reference.selected_cell_count, dtype=np.float64)
    informative_count = 0
    parameters = reference.normalization_parameters
    with threadpool_limits(limits=resources.workers):
        for start in range(0, n_sim, batch_rows):
            stop = min(start + batch_rows, n_sim)
            selected = keep[start:stop]
            if not selected.any():
                continue
            simulated = sum_doublet_pairs(
                selected_pool,
                left[start:stop][selected],
                right[start:stop][selected],
            )
            raw = simulated.toarray()
            del simulated
            normalized = normalize_reference_counts(
                raw,
                size_factor=parameters["size_factor"],
                log_transform=parameters["log_transform"],
                denominator=(
                    None
                    if parameters["renormalize_subset"]
                    else totals[start:stop][selected]
                ),
            )
            del raw
            coordinates = project_pca(normalized, model)
            del normalized
            informative = ~zero_norm_rows(coordinates)
            if informative.any():
                result = query.query(coordinates[informative])
                indices, distances = result[:2]
                weights = mapping_score_weights(distances)
                np.add.at(scores, indices.ravel(), weights.ravel())
                informative_count += int(informative.sum())
                del result, indices, distances, weights
            del coordinates
    if informative_count:
        scores *= 1_000 / (informative_count * n_k)
    np.log1p(scores, out=scores)
    return scores


def smooth_doublet_scores(
    graph: csr_matrix,
    scores: np.ndarray,
    *,
    power: int,
    normalize: bool,
) -> np.ndarray:
    if isinstance(power, bool) or not isinstance(power, int | np.integer):
        raise TypeError("t must be a positive integer")
    if power < 1:
        raise ValueError("t must be a positive integer")
    transition = diffusion_operator(graph, power=1)
    for _ in range(power):
        scores = np.asarray(transition.dot(scores), dtype=np.float64)
    if normalize:
        lo, hi = scores.min(), scores.max()
        scores = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    return scores


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
