"""Paired counts / countsT rotateOnce layout policy."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import zarr

from .layout import _CODEC_MAX_BYTES, ZarrArraySpec, get_compressors
from .profiles import StorageProfile
from .types import array_metadata_shards, as_zarr_array

UNIT_BYTES = 1_000_000_000
CHUNK_BYTES = 100_000_000
COUNT_MATRIX_LAYOUT_KEY = "scarf:countMatrixLayout"
REBUILD_REMEDY = "Rebuild the store with repack_zarr or write_counts_t."


@dataclass(frozen=True, slots=True)
class CountMatrixPolicy:
    """Byte targets that determine paired ``counts`` / ``countsT`` geometry.

    Attributes:
        unitBytes: Target bytes for one destination shard and one
                   all-cell feature read group.
        chunkBytes: Target bytes for one inner Zarr chunk.
    """

    unitBytes: int
    chunkBytes: int

    def __post_init__(self) -> None:
        if min(self.unitBytes, self.chunkBytes) < 1:
            raise ValueError("layout policy values must be positive")
        if self.unitBytes < self.chunkBytes:
            raise ValueError("unitBytes must be at least chunkBytes")

    @property
    def chunksPerShard(self) -> int:
        """Inner chunks packed into one shard from the byte targets."""
        return max(1, (self.unitBytes + self.chunkBytes // 2) // self.chunkBytes)


DEFAULT_COUNT_MATRIX_POLICY = CountMatrixPolicy(
    unitBytes=UNIT_BYTES,
    chunkBytes=CHUNK_BYTES,
)


@dataclass(frozen=True, slots=True)
class CountsTReadGroupSpec:
    featureWidth: int
    cellExtent: int
    chunkFeatures: int
    chunkCells: int
    shardFeatures: int
    shardCells: int
    shardsTouched: int
    chunksTouched: int
    readGroupBytes: int
    physicalShardBytes: int


@dataclass(frozen=True, slots=True)
class CountMatrixPairPlan:
    policy: CountMatrixPolicy
    nCells: int
    nFeats: int
    itemsize: int
    dtype: str
    chunksPerShard: int
    counts: ZarrArraySpec
    countsT: ZarrArraySpec
    readGroup: CountsTReadGroupSpec
    sourceDecodeAmplification: float
    destinationBufferBytes: int
    sourceBufferBytes: int
    fingerprint: str


def _non_negative(value: int, name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _check_codec_limit(chunks: tuple[int, int], itemsize: int, *, name: str) -> None:
    raw = int(chunks[0]) * int(chunks[1]) * int(itemsize)
    if raw > _CODEC_MAX_BYTES:
        raise ValueError(
            f"{name} inner chunk is {raw} bytes, exceeding the codec input "
            f"limit of {_CODEC_MAX_BYTES} bytes"
        )


def _spec(
    shape: tuple[int, int],
    chunks: tuple[int, int],
    shards: tuple[int, int],
    dtype: Any,
    *,
    profile: StorageProfile,
) -> ZarrArraySpec:
    if shards[0] % chunks[0] or shards[1] % chunks[1]:
        raise ValueError(
            f"shards {shards} must be integer multiples of chunks {chunks}"
        )
    return ZarrArraySpec(
        shape=shape,
        chunks=chunks,
        dtype=np.dtype(dtype),
        compressors=get_compressors(profile, zarrFormat=3),
        shards=shards,
        fillValue=0,
        overwrite=True,
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aligned_feature_widths(
    source_limit: int,
    read_limit: int,
    n_feats: int,
) -> tuple[int, int]:
    """Return compatible counts-chunk and all-cell read widths."""
    source_cap = max(1, min(int(n_feats), int(source_limit)))
    read_cap = max(1, min(int(n_feats), int(read_limit)))
    candidates: set[tuple[int, int]] = set()

    if source_cap >= read_cap:
        lower = max(1, source_cap // read_cap)
        upper = max(1, (source_cap + read_cap - 1) // read_cap)
        for ratio in {1, lower, upper}:
            read_width = min(read_cap, source_cap // ratio, n_feats // ratio)
            if read_width > 0:
                candidates.add((ratio * read_width, read_width))
    else:
        lower = max(1, read_cap // source_cap)
        upper = max(1, (read_cap + source_cap - 1) // source_cap)
        for ratio in {1, lower, upper}:
            source_width = min(source_cap, read_cap // ratio, n_feats // ratio)
            if source_width > 0:
                candidates.add((source_width, ratio * source_width))

    def score(widths: tuple[int, int]) -> tuple[int, int, int, int]:
        source_width, read_width = widths
        return (
            min(source_width * read_cap, read_width * source_cap),
            source_width * read_cap + read_width * source_cap,
            read_width,
            source_width,
        )

    return max(candidates, key=score)


def _pad_to_multiple(extent: int, step: int) -> int:
    step = max(1, int(step))
    extent = max(1, int(extent))
    return ((extent + step - 1) // step) * step


def _largest_divisor_at_most(value: int, limit: int) -> int:
    resolved = max(1, int(value))
    ceiling = max(1, min(resolved, int(limit)))
    if ceiling == resolved:
        return resolved
    best = 1
    for divisor in range(1, math.isqrt(resolved) + 1):
        if resolved % divisor:
            continue
        if divisor <= ceiling:
            best = max(best, divisor)
        paired = resolved // divisor
        if paired <= ceiling:
            best = max(best, paired)
    return best


def _empty_plan(
    n_cells: int,
    n_feats: int,
    dtype: Any,
    *,
    policy: CountMatrixPolicy,
    profile: StorageProfile,
) -> CountMatrixPairPlan:
    itemsize = int(np.dtype(dtype).itemsize)
    counts_chunks = (1, 1)
    counts_shards = (1, 1)
    counts_t_chunks = (max(1, n_feats), 1)
    counts_t_shards = (max(1, n_feats), 1)
    counts = _spec(
        (n_cells, n_feats),
        counts_chunks,
        counts_shards,
        dtype,
        profile=profile,
    )
    counts_t = _spec(
        (n_feats, n_cells),
        counts_t_chunks,
        counts_t_shards,
        dtype,
        profile=profile,
    )
    read_group = CountsTReadGroupSpec(
        featureWidth=max(0, n_feats),
        cellExtent=n_cells,
        chunkFeatures=counts_t_chunks[0],
        chunkCells=counts_t_chunks[1],
        shardFeatures=counts_t_shards[0],
        shardCells=counts_t_shards[1],
        shardsTouched=1,
        chunksTouched=1,
        readGroupBytes=0,
        physicalShardBytes=0,
    )
    return _finish_plan(
        policy=policy,
        n_cells=n_cells,
        n_feats=n_feats,
        itemsize=itemsize,
        dtype=dtype,
        counts=counts,
        counts_t=counts_t,
        read_group=read_group,
        amplification=1.0,
        destination_bytes=0,
        source_bytes=0,
    )


def _finish_plan(
    *,
    policy: CountMatrixPolicy,
    n_cells: int,
    n_feats: int,
    itemsize: int,
    dtype: Any,
    counts: ZarrArraySpec,
    counts_t: ZarrArraySpec,
    read_group: CountsTReadGroupSpec,
    amplification: float,
    destination_bytes: int,
    source_bytes: int,
) -> CountMatrixPairPlan:
    counts_t_shards = counts_t.shards or counts_t.chunks
    chunks_per_shard = math.prod(
        max(1, int(shard) // max(1, int(chunk)))
        for shard, chunk in zip(counts_t_shards, counts_t.chunks, strict=True)
    )
    resolved = {
        "policy": {"unitBytes": policy.unitBytes, "chunkBytes": policy.chunkBytes},
        "nCells": n_cells,
        "nFeats": n_feats,
        "itemsize": itemsize,
        "dtype": np.dtype(dtype).name,
        "chunksPerShard": chunks_per_shard,
        "countsChunks": list(counts.chunks),
        "countsShards": list(counts.shards or ()),
        "countsTChunks": list(counts_t.chunks),
        "countsTShards": list(counts_t.shards or ()),
        "readGroup": {
            "featureWidth": read_group.featureWidth,
            "cellExtent": read_group.cellExtent,
            "chunkFeatures": read_group.chunkFeatures,
            "chunkCells": read_group.chunkCells,
            "shardFeatures": read_group.shardFeatures,
            "shardCells": read_group.shardCells,
        },
    }
    return CountMatrixPairPlan(
        policy=policy,
        nCells=n_cells,
        nFeats=n_feats,
        itemsize=itemsize,
        dtype=np.dtype(dtype).name,
        chunksPerShard=chunks_per_shard,
        counts=counts,
        countsT=counts_t,
        readGroup=read_group,
        sourceDecodeAmplification=amplification,
        destinationBufferBytes=destination_bytes,
        sourceBufferBytes=source_bytes,
        fingerprint=_fingerprint(resolved),
    )


def plan_count_matrix_pair(
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    policy: CountMatrixPolicy = DEFAULT_COUNT_MATRIX_POLICY,
    profile: StorageProfile = "cloud",
) -> CountMatrixPairPlan:
    n_cells = _non_negative(nCells, "nCells")
    n_feats = _non_negative(nFeats, "nFeats")
    itemsize = int(np.dtype(dtype).itemsize)
    if itemsize < 1:
        raise ValueError("itemsize must be positive")
    if n_cells == 0 or n_feats == 0:
        return _empty_plan(
            n_cells,
            n_feats,
            dtype,
            policy=policy,
            profile=profile,
        )

    unit_bytes = int(policy.unitBytes)
    chunk_bytes = int(policy.chunkBytes)
    row_bytes = n_feats * itemsize
    counts_shard_cells = max(1, min(n_cells, unit_bytes // max(1, row_bytes)))
    source_feature_limit = max(
        1, min(n_feats, chunk_bytes // max(1, counts_shard_cells * itemsize))
    )
    read_feature_limit = max(1, min(n_feats, unit_bytes // max(1, n_cells * itemsize)))
    counts_chunk_feats, read_feats = _aligned_feature_widths(
        source_feature_limit,
        read_feature_limit,
        n_feats,
    )
    counts_chunks = (counts_shard_cells, counts_chunk_feats)
    counts_shards = (
        counts_shard_cells,
        _pad_to_multiple(n_feats, counts_chunk_feats),
    )

    shard_feats = max(counts_chunk_feats, read_feats)
    source_rows_per_shard = min(
        math.ceil(n_cells / counts_shard_cells),
        max(
            1,
            unit_bytes // max(1, shard_feats * counts_shard_cells * itemsize),
        ),
    )
    shard_cells = source_rows_per_shard * counts_shard_cells
    chunk_feats = read_feats
    chunk_cell_limit = max(
        1,
        chunk_bytes // max(1, chunk_feats * itemsize),
    )
    chunk_cells = _largest_divisor_at_most(shard_cells, chunk_cell_limit)
    counts_t_chunks = (chunk_feats, chunk_cells)
    counts_t_shards = (shard_feats, shard_cells)

    _check_codec_limit(counts_chunks, itemsize, name="counts")
    _check_codec_limit(counts_t_chunks, itemsize, name="countsT")

    counts = _spec(
        (n_cells, n_feats),
        counts_chunks,
        counts_shards,
        dtype,
        profile=profile,
    )
    counts_t = _spec(
        (n_feats, n_cells),
        counts_t_chunks,
        counts_t_shards,
        dtype,
        profile=profile,
    )

    cell_shards = max(1, math.ceil(n_cells / shard_cells))
    gene_shards = max(1, math.ceil(read_feats / shard_feats))
    shards_touched = cell_shards * gene_shards
    chunks_touched = max(1, math.ceil(n_cells / chunk_cells)) * max(
        1, math.ceil(read_feats / chunk_feats)
    )
    read_group = CountsTReadGroupSpec(
        featureWidth=read_feats,
        cellExtent=n_cells,
        chunkFeatures=chunk_feats,
        chunkCells=chunk_cells,
        shardFeatures=shard_feats,
        shardCells=shard_cells,
        shardsTouched=shards_touched,
        chunksTouched=chunks_touched,
        readGroupBytes=n_cells * read_feats * itemsize,
        physicalShardBytes=shard_cells * shard_feats * itemsize,
    )
    destination_bytes = min(shard_cells, n_cells) * min(shard_feats, n_feats) * itemsize
    source_bytes = counts_shard_cells * counts_chunk_feats * itemsize
    return _finish_plan(
        policy=policy,
        n_cells=n_cells,
        n_feats=n_feats,
        itemsize=itemsize,
        dtype=dtype,
        counts=counts,
        counts_t=counts_t,
        read_group=read_group,
        amplification=1.0,
        destination_bytes=destination_bytes,
        source_bytes=source_bytes,
    )


def policy_from_payload(payload: dict[str, Any]) -> CountMatrixPolicy:
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(
            f"count matrix layout metadata is missing a policy. {REBUILD_REMEDY}"
        )
    if "targetReadUnitBytes" in policy or "targetChunkBytes" in policy:
        raise ValueError("count matrix layout uses retired keys. " + REBUILD_REMEDY)
    if "unitBytes" not in policy or "chunkBytes" not in policy:
        raise ValueError(
            f"count matrix layout metadata is incomplete. {REBUILD_REMEDY}"
        )
    return CountMatrixPolicy(
        unitBytes=int(policy["unitBytes"]),
        chunkBytes=int(policy["chunkBytes"]),
    )


def replay_count_matrix_plan(
    payload: dict[str, Any],
    *,
    nCells: int,
    nFeats: int,
    dtype: Any,
    profile: StorageProfile = "cloud",
) -> CountMatrixPairPlan:
    policy = policy_from_payload(payload)
    return plan_count_matrix_pair(
        nCells,
        nFeats,
        dtype,
        policy=policy,
        profile=profile,
    )


def validate_count_matrix_pair(
    plan: CountMatrixPairPlan,
    *,
    expected: CountMatrixPairPlan,
) -> None:
    if plan.fingerprint != expected.fingerprint:
        raise ValueError("count matrix plan fingerprint does not match expected policy")
    if plan.counts.shape != expected.counts.shape:
        raise ValueError("counts shape mismatch")
    if plan.countsT.shape != expected.countsT.shape:
        raise ValueError("countsT shape mismatch")
    if plan.counts.chunks != expected.counts.chunks:
        raise ValueError("counts chunks mismatch")
    if plan.counts.shards != expected.counts.shards:
        raise ValueError("counts shards mismatch")
    if plan.countsT.chunks != expected.countsT.chunks:
        raise ValueError("countsT chunks mismatch")
    if plan.countsT.shards != expected.countsT.shards:
        raise ValueError("countsT shards mismatch")
    if np.dtype(plan.counts.dtype) != np.dtype(expected.counts.dtype):
        raise ValueError("counts dtype mismatch")


def _array_geometry(
    array: zarr.Array,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]:
    stored = array_metadata_shards(array)
    return (
        tuple(int(value) for value in array.shape),
        tuple(int(value) for value in array.chunks),
        None if stored is None else tuple(int(value) for value in stored),
    )


def validate_count_matrix_source(
    counts: zarr.Array,
    *,
    expected: CountMatrixPairPlan,
) -> None:
    """Validate a counts array against the paired plan and persisted payload."""
    shape, chunks, shards = _array_geometry(counts)
    if shape != expected.counts.shape:
        raise ValueError("counts shape does not match the paired plan")
    if np.dtype(counts.dtype) != np.dtype(expected.counts.dtype):
        raise ValueError("counts dtype does not match the paired plan")
    if chunks != expected.counts.chunks:
        raise ValueError("counts chunks do not match the paired plan")
    if shards is None or shards != expected.counts.shards:
        raise ValueError("counts shards do not match the paired plan")
    recorded = load_count_matrix_plan(counts)
    if recorded.get("fingerprint") != expected.fingerprint:
        raise ValueError(
            "counts metadata does not match the expected paired plan. " + REBUILD_REMEDY
        )


def validate_live_count_matrix_geometry(
    array: zarr.Array,
    *,
    expected_shape: tuple[int, ...],
    expected_chunks: tuple[int, ...],
    expected_shards: tuple[int, ...] | None,
    dtype: Any,
    name: str,
) -> None:
    shape, chunks, shards = _array_geometry(array)
    if shape != tuple(int(value) for value in expected_shape):
        raise ValueError(f"{name} shape does not match the persisted plan")
    if np.dtype(array.dtype) != np.dtype(dtype):
        raise ValueError(f"{name} dtype does not match the persisted plan")
    if chunks != tuple(int(value) for value in expected_chunks):
        raise ValueError(f"{name} chunks do not match the persisted plan")
    if shards != (
        None
        if expected_shards is None
        else tuple(int(value) for value in expected_shards)
    ):
        raise ValueError(f"{name} shards do not match the persisted plan")


def create_count_matrix_array(group: Any, name: str, spec: ZarrArraySpec) -> zarr.Array:
    """Create an array using the planned chunk/shard grid without clamping."""
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": spec.chunks,
        "dtype": spec.dtype,
        "compressors": spec.compressors,
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None:
        kwargs["shards"] = spec.shards
    if spec.fillValue is not None:
        kwargs["fill_value"] = spec.fillValue
    return as_zarr_array(group.create_array(name, **kwargs), name=name)


def persist_count_matrix_plan(group: Any, plan: CountMatrixPairPlan) -> None:
    payload = {
        "policy": {
            "unitBytes": plan.policy.unitBytes,
            "chunkBytes": plan.policy.chunkBytes,
        },
        "nCells": plan.nCells,
        "nFeats": plan.nFeats,
        "dtype": plan.dtype,
        "itemsize": plan.itemsize,
        "chunksPerShard": plan.chunksPerShard,
        "counts": {
            "shape": list(plan.counts.shape),
            "chunks": list(plan.counts.chunks),
            "shards": None if plan.counts.shards is None else list(plan.counts.shards),
        },
        "countsT": {
            "shape": list(plan.countsT.shape),
            "chunks": list(plan.countsT.chunks),
            "shards": None
            if plan.countsT.shards is None
            else list(plan.countsT.shards),
        },
        "readGroup": asdict(plan.readGroup),
        "sourceDecodeAmplification": plan.sourceDecodeAmplification,
        "fingerprint": plan.fingerprint,
    }
    group.attrs[COUNT_MATRIX_LAYOUT_KEY] = payload


def load_count_matrix_plan(group: Any) -> dict[str, Any]:
    payload = group.attrs.get(COUNT_MATRIX_LAYOUT_KEY)
    if not isinstance(payload, dict):
        raise ValueError(f"count matrix layout metadata is missing. {REBUILD_REMEDY}")
    policy_from_payload(payload)
    return dict(payload)


def require_matching_count_matrix_plans(
    *groups: Any,
) -> dict[str, Any]:
    payloads = [load_count_matrix_plan(group) for group in groups]
    fingerprints = {payload.get("fingerprint") for payload in payloads}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(
            "count matrix layout metadata does not agree across anchors. "
            + REBUILD_REMEDY
        )
    return payloads[0]


def read_group_from_payload(payload: dict[str, Any]) -> tuple[int, int]:
    """Return persisted ``(featureWidth, readGroupBytes)`` or raise."""
    read_group = payload.get("readGroup")
    if (
        not isinstance(read_group, dict)
        or "featureWidth" not in read_group
        or "readGroupBytes" not in read_group
    ):
        raise ValueError(
            "count matrix layout is missing a persisted read group. " + REBUILD_REMEDY
        )
    feature_width = int(read_group["featureWidth"])
    read_group_bytes = int(read_group["readGroupBytes"])
    if feature_width < 0 or read_group_bytes < 0:
        raise ValueError(
            "count matrix layout has an invalid persisted read group. " + REBUILD_REMEDY
        )
    return feature_width, read_group_bytes


def require_count_matrix_layout(
    group: Any,
    counts: Any,
    counts_t: Any | None = None,
    *,
    profile: StorageProfile = "cloud",
) -> CountMatrixPairPlan:
    """Load, agree, and replay persisted layout metadata against live arrays."""
    anchors = [group, counts]
    if counts_t is not None:
        anchors.append(counts_t)
    payload = require_matching_count_matrix_plans(*anchors)
    plan = replay_count_matrix_plan(
        payload,
        nCells=int(counts.shape[0]),
        nFeats=int(counts.shape[1]),
        dtype=counts.dtype,
        profile=profile,
    )
    if payload.get("fingerprint") != plan.fingerprint:
        raise ValueError(
            "persisted count matrix plan does not replay against live "
            f"geometry. {REBUILD_REMEDY}"
        )
    feature_width, read_group_bytes = read_group_from_payload(payload)
    if feature_width != int(plan.readGroup.featureWidth) or read_group_bytes != int(
        plan.readGroup.readGroupBytes
    ):
        raise ValueError(
            "persisted read group does not match live geometry. " + REBUILD_REMEDY
        )
    validate_live_count_matrix_geometry(
        counts,
        expected_shape=plan.counts.shape,
        expected_chunks=plan.counts.chunks,
        expected_shards=plan.counts.shards,
        dtype=plan.counts.dtype,
        name="counts",
    )
    if counts_t is not None:
        validate_live_count_matrix_geometry(
            counts_t,
            expected_shape=plan.countsT.shape,
            expected_chunks=plan.countsT.chunks,
            expected_shards=plan.countsT.shards,
            dtype=plan.countsT.dtype,
            name="countsT",
        )
    return plan


def create_product_counts_array(
    group: Any,
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    profile: StorageProfile,
    policy: CountMatrixPolicy | None = None,
    zarrFormat: int = 3,
) -> zarr.Array:
    """Create a ``counts`` array using the rotateOnce product layout."""
    if int(zarrFormat) < 3:
        raise ValueError("paired count matrices require Zarr format 3")
    plan = plan_count_matrix_pair(
        nCells,
        nFeats,
        dtype,
        policy=policy or DEFAULT_COUNT_MATRIX_POLICY,
        profile=profile,
    )
    counts = create_count_matrix_array(group, "counts", plan.counts)
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    return as_zarr_array(counts, name="counts")
