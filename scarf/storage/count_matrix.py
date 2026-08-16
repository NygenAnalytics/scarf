"""Paired counts / countsT layout policy for the experimental geometry.

This module is additive. Public writers keep the current planners until a
later phase selects this policy as the product path.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import zarr

from .layout import _CODEC_MAX_BYTES, ZarrArraySpec, get_compressors
from .profiles import StorageProfile
from .types import array_metadata_shards, as_zarr_array

TARGET_READ_UNIT_BYTES = 1_000_000_000
TARGET_CHUNK_BYTES = 100_000_000

LayoutStrategy = Literal["keepAspect", "rotateEach", "rotateOnce"]
LAYOUT_STRATEGIES: tuple[LayoutStrategy, ...] = (
    "keepAspect",
    "rotateEach",
    "rotateOnce",
)
# Locked after the scaled 1M/5M write/read comparison. keepAspect and
# rotateOnce are identical through 1M cells. Past that, rotateOnce keeps
# consumer read groups near U with a 10x write-decode cap. rotateEach is
# kept as a planner variant only; its 100x write cost at 100M is not the
# experimental default.
DEFAULT_LAYOUT_STRATEGY: LayoutStrategy = "rotateOnce"


@dataclass(frozen=True, slots=True)
class CountMatrixLayoutPolicy:
    targetReadUnitBytes: int
    targetChunkBytes: int

    def __post_init__(self) -> None:
        if min(self.targetReadUnitBytes, self.targetChunkBytes) < 1:
            raise ValueError("layout policy values must be positive")
        if self.targetReadUnitBytes < self.targetChunkBytes:
            raise ValueError("targetReadUnitBytes must be at least targetChunkBytes")

    @property
    def chunksPerShard(self) -> int:
        return max(
            1,
            (self.targetReadUnitBytes + self.targetChunkBytes // 2)
            // self.targetChunkBytes,
        )


EXPERIMENTAL_POLICY = CountMatrixLayoutPolicy(
    targetReadUnitBytes=TARGET_READ_UNIT_BYTES,
    targetChunkBytes=TARGET_CHUNK_BYTES,
)

# Product writes stay on the current layout until a later phase records a switch.
ACCEPTED_LAYOUT_BRANCH: str = "current"
_EXPERIMENTAL_LAYOUT_READS: ContextVar[bool] = ContextVar(
    "scarf_experimental_layout_reads",
    default=False,
)


def accepted_layout_branch() -> str:
    """Return the recorded product-layout branch, or ``current`` before a switch."""
    return ACCEPTED_LAYOUT_BRANCH


def apply_recorded_layout_branch(branch: str) -> str:
    """Map a recorded experiment branch onto the product layout switch.

    Only Branch A replaces the on-disk counts/countsT geometry. Every other
    recorded outcome keeps the current layout.
    """
    global ACCEPTED_LAYOUT_BRANCH
    if branch == "A":
        ACCEPTED_LAYOUT_BRANCH = "A"
    elif branch in {"B", "C", "D", "E", "current"}:
        ACCEPTED_LAYOUT_BRANCH = "current"
    else:
        raise ValueError(f"unsupported layout branch {branch!r}")
    return ACCEPTED_LAYOUT_BRANCH


@contextmanager
def override_accepted_layout_branch(branch: str) -> Iterator[str]:
    """Temporarily apply a recorded branch, then restore the previous value."""
    global ACCEPTED_LAYOUT_BRANCH
    previous = ACCEPTED_LAYOUT_BRANCH
    try:
        yield apply_recorded_layout_branch(branch)
    finally:
        ACCEPTED_LAYOUT_BRANCH = previous


def uses_experimental_product_layout() -> bool:
    return accepted_layout_branch() == "A"


def experimental_layout_reads_enabled() -> bool:
    """Return whether the current experimental operation may open paired arrays."""
    return _EXPERIMENTAL_LAYOUT_READS.get()


@contextmanager
def allow_experimental_layout_reads() -> Iterator[None]:
    """Temporarily allow private experiment code to open paired countsT arrays."""
    token = _EXPERIMENTAL_LAYOUT_READS.set(True)
    try:
        yield
    finally:
        _EXPERIMENTAL_LAYOUT_READS.reset(token)


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
    policy: CountMatrixLayoutPolicy
    strategy: str
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


def _positive(value: int, name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _nearest_divisor(value: int, target: int, *, multipleOf: int = 1) -> int:
    resolved = max(1, int(value))
    wanted = min(max(1, int(target)), resolved)
    step = max(1, int(multipleOf))
    best = 1
    best_key = (abs(1 - wanted), 1)
    candidate = 1
    while candidate * candidate <= resolved:
        if resolved % candidate == 0:
            for item in (candidate, resolved // candidate):
                if item != resolved and step > 1 and item % step:
                    continue
                key = (abs(item - wanted), item)
                if key < best_key:
                    best = item
                    best_key = key
        candidate += 1
    if best == 1 and resolved > 1 and wanted > 1:
        return resolved
    return best


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


def _factor_k(k: int, ratio: int) -> tuple[int, int]:
    """Return ``(cellParts, geneParts)`` whose product is ``k``."""
    target = min(max(1, int(k)), max(1, int(ratio)))
    gene_parts = int(k)
    for divisor in range(target, int(k) + 1):
        if int(k) % divisor == 0:
            gene_parts = divisor
            break
    return int(k) // gene_parts, gene_parts


def _pad_to_multiple(extent: int, step: int) -> int:
    step = max(1, int(step))
    extent = max(1, int(extent))
    return ((extent + step - 1) // step) * step


def _counts_t_aspect(
    n_cells: int,
    n_feats: int,
    counts_chunk_feats: int,
    itemsize: int,
    *,
    unit_bytes: int,
    k: int,
    strategy: LayoutStrategy,
) -> tuple[int, int, int]:
    base_cells = max(1, unit_bytes // max(1, counts_chunk_feats * itemsize))
    if n_cells <= base_cells:
        shard_cells = n_cells
        raw_feats = max(1, unit_bytes // max(1, n_cells * itemsize))
        shard_feats = min(n_feats, raw_feats)
        shard_feats = _nearest_divisor(n_feats, shard_feats)
        return shard_cells, shard_feats, 0

    shard_cells = base_cells
    shard_feats = counts_chunk_feats
    rotations = 0
    if strategy == "keepAspect":
        max_rotations = 0
    elif strategy == "rotateOnce":
        max_rotations = 1
    else:
        max_rotations = 64
    while (
        rotations < max_rotations
        and n_cells > k * shard_cells
        and shard_feats >= k
        and shard_feats % k == 0
    ):
        shard_cells *= k
        shard_feats //= k
        rotations += 1
    return shard_cells, shard_feats, rotations


def _counts_t_chunks(
    n_cells: int,
    shard_cells: int,
    shard_feats: int,
    *,
    unit_bytes: int,
    chunk_bytes: int,
    k: int,
    itemsize: int,
) -> tuple[int, int]:
    ratio = max(1, math.ceil(n_cells / max(1, shard_cells)))
    if ratio <= k and shard_feats % max(1, _factor_k(k, ratio)[1]) == 0:
        cell_parts, gene_parts = _factor_k(k, ratio)
        if (
            shard_cells % max(1, cell_parts) == 0
            and shard_feats % max(1, gene_parts) == 0
        ):
            chunk_cells = max(1, shard_cells // cell_parts)
            chunk_feats = max(1, shard_feats // gene_parts)
            return chunk_feats, chunk_cells

    target_feats = max(1, unit_bytes // max(1, n_cells * itemsize))
    chunk_feats = _nearest_divisor(shard_feats, min(shard_feats, target_feats))
    chunk_cells = shard_cells
    nominal = chunk_feats * chunk_cells * itemsize
    if nominal > chunk_bytes * 2 and shard_cells > 1:
        cell_target = max(1, chunk_bytes // max(1, chunk_feats * itemsize))
        chunk_cells = _nearest_divisor(shard_cells, min(shard_cells, cell_target))
    return chunk_feats, chunk_cells


def plan_count_matrix_pair(
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    policy: CountMatrixLayoutPolicy = EXPERIMENTAL_POLICY,
    strategy: LayoutStrategy = DEFAULT_LAYOUT_STRATEGY,
    profile: StorageProfile = "cloud",
) -> CountMatrixPairPlan:
    if strategy not in LAYOUT_STRATEGIES:
        raise ValueError(f"unsupported layout strategy {strategy!r}")
    n_cells = _positive(nCells, "nCells")
    n_feats = _positive(nFeats, "nFeats")
    itemsize = int(np.dtype(dtype).itemsize)
    if itemsize < 1:
        raise ValueError("itemsize must be positive")

    unit_bytes = int(policy.targetReadUnitBytes)
    chunk_bytes = int(policy.targetChunkBytes)
    k = policy.chunksPerShard
    row_bytes = n_feats * itemsize
    counts_shard_cells = max(1, min(n_cells, unit_bytes // max(1, row_bytes)))
    target_chunk_feats = max(1, chunk_bytes // max(1, counts_shard_cells * itemsize))
    counts_chunk_feats = _nearest_divisor(
        n_feats,
        min(n_feats, target_chunk_feats),
        multipleOf=k if n_feats % k == 0 else 1,
    )
    counts_chunks = (counts_shard_cells, counts_chunk_feats)
    counts_shards = (counts_shard_cells, n_feats)

    shard_cells, shard_feats, _rotations = _counts_t_aspect(
        n_cells,
        n_feats,
        counts_chunk_feats,
        itemsize,
        unit_bytes=unit_bytes,
        k=k,
        strategy=strategy,
    )
    chunk_feats, chunk_cells = _counts_t_chunks(
        n_cells,
        shard_cells,
        shard_feats,
        unit_bytes=unit_bytes,
        chunk_bytes=chunk_bytes,
        k=k,
        itemsize=itemsize,
    )
    if shard_cells % chunk_cells:
        shard_cells = _pad_to_multiple(shard_cells, chunk_cells)
    if shard_feats % chunk_feats:
        shard_feats = _pad_to_multiple(shard_feats, chunk_feats)
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

    read_feats = min(n_feats, max(1, unit_bytes // max(1, n_cells * itemsize)))
    read_feats = _nearest_divisor(n_feats, read_feats)
    if read_feats > shard_feats:
        read_feats = min(n_feats, max(chunk_feats, shard_feats))
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
    amplification = max(1.0, counts_chunk_feats / max(1, shard_feats))
    destination_bytes = min(shard_cells, n_cells) * min(shard_feats, n_feats) * itemsize
    source_bytes = counts_shard_cells * counts_chunk_feats * itemsize
    resolved = {
        "policy": asdict(policy),
        "strategy": strategy,
        "nCells": n_cells,
        "nFeats": n_feats,
        "itemsize": itemsize,
        "dtype": np.dtype(dtype).name,
        "chunksPerShard": k,
        "countsChunks": list(counts.chunks),
        "countsShards": list(counts.shards or ()),
        "countsTChunks": list(counts_t.chunks),
        "countsTShards": list(counts_t.shards or ()),
        "readGroup": asdict(read_group),
        "sourceDecodeAmplification": amplification,
    }
    return CountMatrixPairPlan(
        policy=policy,
        strategy=strategy,
        nCells=n_cells,
        nFeats=n_feats,
        itemsize=itemsize,
        dtype=np.dtype(dtype).name,
        chunksPerShard=k,
        counts=counts,
        countsT=counts_t,
        readGroup=read_group,
        sourceDecodeAmplification=amplification,
        destinationBufferBytes=destination_bytes,
        sourceBufferBytes=source_bytes,
        fingerprint=_fingerprint(resolved),
    )


def plan_layout_candidates(
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    policy: CountMatrixLayoutPolicy = EXPERIMENTAL_POLICY,
    profile: StorageProfile = "cloud",
) -> dict[str, CountMatrixPairPlan]:
    return {
        strategy: plan_count_matrix_pair(
            nCells,
            nFeats,
            dtype,
            policy=policy,
            strategy=strategy,
            profile=profile,
        )
        for strategy in LAYOUT_STRATEGIES
    }


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


def validate_count_matrix_source(
    counts: zarr.Array,
    *,
    expected: CountMatrixPairPlan,
) -> None:
    """Validate an experimental counts array and its persisted paired plan."""
    actual_shards = array_metadata_shards(counts)
    if tuple(int(value) for value in counts.shape) != expected.counts.shape:
        raise ValueError("experimental counts shape does not match the paired plan")
    if np.dtype(counts.dtype) != np.dtype(expected.counts.dtype):
        raise ValueError("experimental counts dtype does not match the paired plan")
    if tuple(int(value) for value in counts.chunks) != expected.counts.chunks:
        raise ValueError("experimental counts chunks do not match the paired plan")
    if (
        actual_shards is None
        or tuple(int(value) for value in actual_shards) != expected.counts.shards
    ):
        raise ValueError("experimental counts shards do not match the paired plan")
    recorded = load_count_matrix_plan(counts)
    if recorded.get("fingerprint") != expected.fingerprint:
        raise ValueError(
            "experimental counts metadata does not match the expected paired plan"
        )


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
        "policy": asdict(plan.policy),
        "strategy": plan.strategy,
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
    group.attrs["scarf:countMatrixLayout"] = payload


def load_count_matrix_plan(group: Any) -> dict[str, Any]:
    payload = group.attrs.get("scarf:countMatrixLayout")
    if not isinstance(payload, dict):
        raise ValueError("experimental count matrix layout metadata is missing")
    return dict(payload)


def product_counts_spec(
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    profile: StorageProfile,
    targetChunkBytes: int | None = None,
    targetShardBytes: int | None = None,
    zarrFormat: int = 3,
) -> ZarrArraySpec:
    """Return the counts spec for a new write under the accepted branch."""
    if uses_experimental_product_layout() and int(zarrFormat) >= 3:
        return plan_count_matrix_pair(nCells, nFeats, dtype, profile=profile).counts
    from .layout import count_array_spec

    return count_array_spec(
        nCells,
        nFeats,
        dtype,
        profile=profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
        zarrFormat=zarrFormat,
    )


def create_product_counts_array(
    group: Any,
    nCells: int,
    nFeats: int,
    dtype: Any,
    *,
    profile: StorageProfile,
    targetChunkBytes: int | None = None,
    targetShardBytes: int | None = None,
    zarrFormat: int = 3,
) -> zarr.Array:
    """Create a ``counts`` array using the accepted product layout."""
    if uses_experimental_product_layout() and int(zarrFormat) >= 3:
        plan = plan_count_matrix_pair(
            nCells,
            nFeats,
            dtype,
            profile=profile,
        )
        counts = create_count_matrix_array(group, "counts", plan.counts)
        persist_count_matrix_plan(group, plan)
        return as_zarr_array(counts, name="counts")
    from .arrays import create_numeric_array
    from .layout import count_array_spec

    spec = count_array_spec(
        nCells,
        nFeats,
        dtype,
        profile=profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
        zarrFormat=zarrFormat,
    )
    return create_numeric_array(group, "counts", spec)


def reject_noncanonical_write_destination(group: Any) -> None:
    """Reject old layouts when the accepted branch requires paired metadata."""
    if not uses_experimental_product_layout():
        return
    try:
        load_count_matrix_plan(group)
    except ValueError as exc:
        raise ValueError(
            "This operation requires the accepted paired counts/countsT layout. "
            "Repack the store with repack_zarr before writing."
        ) from exc
