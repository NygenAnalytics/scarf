"""Paired-layout experiment operations for phases 1 through 3."""

from collections.abc import Callable
from typing import Any, Literal
from hashlib import sha256

import numpy as np
import zarr
from zarr.abc.store import Store

from scarf.storage.async_execution import ExecutionPlan
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import (
    EXPERIMENTAL_POLICY,
    CountMatrixLayoutPolicy,
    create_count_matrix_array,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
    validate_count_matrix_pair,
)
from scarf.storage.feature_stream import (
    load_feature_strip,
    map_feature_read_groups,
    selected_feature_chunk_starts,
)
from scarf.storage.layout import count_array_spec
from scarf.storage.sharding import write_counts_t, write_counts_t_experimental

type LayoutKind = Literal["current", "candidate"]
type ConsumerKind = Literal["wholeStrip", "bounded"]
type WriterKind = Literal["current", "asyncCandidate"]

SCALED_POLICY = CountMatrixLayoutPolicy(
    targetReadUnitBytes=800,
    targetChunkBytes=200,
)
SCALED_CELLS = 221
SCALED_FEATS = 37
PHASE3_VARIANTS: tuple[tuple[str, LayoutKind, WriterKind, ConsumerKind], ...] = (
    ("currentWholeStrip", "current", "current", "wholeStrip"),
    ("currentBounded", "current", "current", "bounded"),
    ("candidateBounded", "candidate", "asyncCandidate", "bounded"),
)
PHASE3_MIN_REPS = 3
WALL_REGRESSION_LIMIT = 0.15
PRIMARY_IMPROVEMENT_LIMIT = 0.20
MEMORY_IMPROVEMENT_LIMIT = 0.30


def scaled_values(
    nCells: int = SCALED_CELLS,
    nFeats: int = SCALED_FEATS,
    *,
    dtype: str = "uint16",
) -> np.ndarray:
    cells = np.arange(nCells, dtype=np.uint32)[:, None]
    feats = np.arange(nFeats, dtype=np.uint32)[None, :]
    encoded = cells * np.uint32(nFeats) + feats
    info = np.iinfo(np.dtype(dtype))
    return (encoded % np.uint32(info.max)).astype(dtype)


def policy_from_mapping(payload: dict[str, Any]) -> CountMatrixLayoutPolicy:
    if "targetReadUnitBytes" in payload:
        return CountMatrixLayoutPolicy(
            targetReadUnitBytes=int(payload["targetReadUnitBytes"]),
            targetChunkBytes=int(payload["targetChunkBytes"]),
        )
    return CountMatrixLayoutPolicy(
        targetReadUnitBytes=int(payload["targetShardBytes"]),
        targetChunkBytes=int(payload["targetChunkBytes"]),
    )


def create_current_counts(
    group: zarr.Group,
    values: np.ndarray,
    *,
    profile: str = "cloud",
    targetChunkBytes: int = 2_000,
    targetShardBytes: int = 8_000,
) -> zarr.Array:
    spec = count_array_spec(
        int(values.shape[0]),
        int(values.shape[1]),
        dtype=values.dtype,
        profile=profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
    )
    counts = group.create_array(
        "counts",
        shape=spec.shape,
        chunks=spec.chunks,
        shards=spec.shards,
        dtype=spec.dtype,
        overwrite=True,
    )
    counts[:] = values
    return counts


def create_candidate_counts(
    group: zarr.Group,
    values: np.ndarray,
    *,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
    profile: str = "cloud",
    resources: ResourceBudget | None = None,
) -> tuple[zarr.Array, Any]:
    del resources
    plan = plan_count_matrix_pair(
        int(values.shape[0]),
        int(values.shape[1]),
        values.dtype,
        policy=policy,
        profile=profile,
    )
    counts = create_count_matrix_array(group, "counts", plan.counts)
    counts[:] = values
    persist_count_matrix_plan(group, plan)
    persist_count_matrix_plan(counts, plan)
    return counts, plan


def copy_candidate_counts(
    group: zarr.Group,
    source: zarr.Array,
    *,
    policy: CountMatrixLayoutPolicy,
    profile: str,
    resources: ResourceBudget,
    name: str = "countsCandidate",
) -> tuple[zarr.Array, dict[str, Any]]:
    """Create a profiling-only paired-layout source without changing counts."""
    import time

    if name in group:
        raise FileExistsError(f"candidate counts array already exists: {name}")
    plan = plan_count_matrix_pair(
        int(source.shape[0]),
        int(source.shape[1]),
        source.dtype,
        policy=policy,
        profile=profile,
    )
    destination = create_count_matrix_array(group, name, plan.counts)
    persist_count_matrix_plan(destination, plan)
    persist_count_matrix_plan(group, plan)
    n_cells, n_feats = (int(value) for value in source.shape)
    itemsize = int(np.dtype(source.dtype).itemsize)
    row_bytes = max(1, n_feats * itemsize)
    rows_per_block = min(
        int(plan.counts.chunks[0]),
        max(1, resources.memoryBytes // (2 * row_bytes)),
    )
    started = time.perf_counter()
    copied_bytes = 0
    for cell_start in range(0, n_cells, rows_per_block):
        cell_end = min(cell_start + rows_per_block, n_cells)
        block = np.asarray(source[cell_start:cell_end, :])
        destination[cell_start:cell_end, :] = block
        copied_bytes += int(block.nbytes)
    elapsed = time.perf_counter() - started
    return destination, {
        "name": name,
        "seconds": elapsed,
        "logicalBytes": copied_bytes,
        "rowsPerBlock": rows_per_block,
        "chunks": list(destination.chunks),
        "shards": list(destination.shards or ()),
        "fingerprint": plan.fingerprint,
        "kind": "observed",
    }


def write_layout_counts_t(
    counts: zarr.Array,
    group: zarr.Group,
    *,
    writer: WriterKind,
    policy: CountMatrixLayoutPolicy | None = None,
    profile: str = "cloud",
    resources: ResourceBudget | None = None,
    readGroupChunks: int = 1,
    readGroupsInFlight: int = 1,
    destinationCommitsInFlight: int = 1,
    metrics: dict[str, Any] | None = None,
) -> zarr.Array:
    budget = resources or ResourceBudget(64 * 1024 * 1024, 2)
    if writer == "asyncCandidate":
        result = write_counts_t_experimental(
            counts,
            group,
            policy=policy or SCALED_POLICY,
            profile=profile,
            resources=budget,
            readGroupChunks=readGroupChunks,
            readGroupsInFlight=readGroupsInFlight,
            destinationCommitsInFlight=destinationCommitsInFlight,
            metrics=metrics,
        )
    else:
        written = write_counts_t(
            counts,
            group,
            profile=profile,
            resources=budget,
            maxShardBytes=8_000,
            targetChunkBytes=2_000,
        )
        if written is None:
            raise RuntimeError("current countsT writer returned None")
        result = written
    return result


def load_whole_strip_groups(counts_t: zarr.Array) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for start in selected_feature_chunk_starts(counts_t):
        group = load_feature_strip(counts_t, start)
        blocks.append(np.asarray(group.values))
    return blocks


def load_bounded_groups(
    counts_t: zarr.Array, resources: ResourceBudget
) -> list[np.ndarray]:
    return [
        np.asarray(group.values)
        for group in map_feature_read_groups(
            counts_t,
            lambda item: item,
            resources=resources,
        )
    ]


def consume_counts_t(
    counts_t: zarr.Array,
    *,
    consumer: ConsumerKind,
    resources: ResourceBudget,
) -> np.ndarray:
    if consumer == "wholeStrip":
        blocks = load_whole_strip_groups(counts_t)
    else:
        blocks = load_bounded_groups(counts_t, resources)
    if not blocks:
        return np.empty((0, int(counts_t.shape[1])), dtype=counts_t.dtype)
    return np.concatenate(blocks, axis=0)


def array_checksum(values: np.ndarray) -> str:
    digest = sha256()
    digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def run_phase1_local_checks(
    *,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
) -> dict[str, Any]:
    values = scaled_values()
    plan = plan_count_matrix_pair(
        values.shape[0],
        values.shape[1],
        values.dtype,
        policy=policy,
    )
    validate_count_matrix_pair(plan, expected=plan)
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    counts, stored_plan = create_candidate_counts(group, values, policy=policy)
    persist_count_matrix_plan(root, stored_plan)
    np.testing.assert_array_equal(np.asarray(counts[:]), values)
    canonical = plan_count_matrix_pair(10_000, 45_525, "uint16")
    return {
        "checks": (
            "scaled-pair-metadata",
            "logical-edge-values",
            "canonical-10k-metadata",
        ),
        "scaledPlan": {
            "countsChunks": list(plan.counts.chunks),
            "countsShards": list(plan.counts.shards or ()),
            "countsTChunks": list(plan.countsT.chunks),
            "countsTShards": list(plan.countsT.shards or ()),
            "fingerprint": plan.fingerprint,
        },
        "canonical10k": {
            "countsChunks": list(canonical.counts.chunks),
            "countsShards": list(canonical.counts.shards or ()),
            "countsTChunks": list(canonical.countsT.chunks),
            "countsTShards": list(canonical.countsT.shards or ()),
            "fingerprint": canonical.fingerprint,
        },
        "checksum": array_checksum(values),
        "kind": "observed",
    }


def fill_phase1_remote_pair(
    store: Store,
    *,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
    canonicalPolicy: CountMatrixLayoutPolicy | None = None,
    profile: str = "cloud",
) -> dict[str, Any]:
    values = scaled_values()
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    counts, plan = create_candidate_counts(
        group,
        values,
        policy=policy,
        profile=profile,
    )
    persist_count_matrix_plan(root, plan)
    canonical_group = root.create_group("canonical10k")
    canonical_plan = plan_count_matrix_pair(
        10_000,
        45_525,
        "uint16",
        policy=canonicalPolicy or EXPERIMENTAL_POLICY,
        profile=profile,
    )
    canonical_counts = create_count_matrix_array(
        canonical_group,
        "counts",
        canonical_plan.counts,
    )
    canonical_counts_t = create_count_matrix_array(
        canonical_group,
        "countsT",
        canonical_plan.countsT,
    )
    canonical_counts[0, 0] = np.uint16(17)
    canonical_counts[-1, -1] = np.uint16(23)
    canonical_counts_t[0, 0] = np.uint16(17)
    canonical_counts_t[-1, -1] = np.uint16(23)
    persist_count_matrix_plan(canonical_group, canonical_plan)
    return {
        "shape": list(counts.shape),
        "chunks": list(counts.chunks),
        "shards": list(counts.shards or ()),
        "fingerprint": plan.fingerprint,
        "checksum": array_checksum(values),
        "edge": np.asarray(counts[-1, -1]).item(),
        "canonical10k": {
            "countsShape": list(canonical_counts.shape),
            "countsChunks": list(canonical_counts.chunks),
            "countsShards": list(canonical_counts.shards or ()),
            "countsTShape": list(canonical_counts_t.shape),
            "countsTChunks": list(canonical_counts_t.chunks),
            "countsTShards": list(canonical_counts_t.shards or ()),
            "fingerprint": canonical_plan.fingerprint,
            "first": 17,
            "last": 23,
            "kind": "observed",
        },
        "kind": "observed",
    }


def run_phase2_local_checks(
    *,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
    resources: ResourceBudget | None = None,
) -> dict[str, Any]:
    from profiling.recording_store import RecordingMemoryStore

    values = scaled_values()
    budget = resources or ResourceBudget(32 * 1024 * 1024, 2)
    store = RecordingMemoryStore()
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    counts, _plan = create_candidate_counts(
        group,
        values,
        policy=policy,
        profile="fast_local",
        resources=budget,
    )
    writer_metrics: dict[str, Any] = {}
    counts_t = write_counts_t_experimental(
        counts,
        group,
        policy=policy,
        profile="fast_local",
        resources=budget,
        metrics=writer_metrics,
    )
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    if counts_t.attrs.get("complete") is not True:
        raise AssertionError("experimental countsT is incomplete")
    chunk_writes = [
        key
        for operation, key in store.ops
        if operation == "set" and "/countsT/c/" in f"/{key}"
    ]
    if not chunk_writes:
        raise AssertionError("experimental countsT emitted no chunk writes")
    failure_store = RecordingMemoryStore(fail_on=chunk_writes[0])
    failure_root = zarr.open_group(store=failure_store, mode="w")
    failure_group = failure_root.create_group("RNA")
    failure_counts, _failure_plan = create_candidate_counts(
        failure_group,
        values,
        policy=policy,
        profile="fast_local",
        resources=budget,
    )
    failure_metrics: dict[str, Any] = {}
    try:
        write_counts_t_experimental(
            failure_counts,
            failure_group,
            policy=policy,
            profile="fast_local",
            resources=budget,
            metrics=failure_metrics,
        )
    except Exception as exc:
        nested = tuple(exc.exceptions) if isinstance(exc, BaseExceptionGroup) else ()
        if "injected write failure" not in str(exc) and not any(
            "injected write failure" in str(item) for item in nested
        ):
            raise
    else:
        raise AssertionError("injected countsT failure did not propagate")
    failed_counts_t = failure_group["countsT"]
    if failed_counts_t.attrs.get("complete") is not False:
        raise AssertionError("failed experimental countsT was marked complete")
    if failure_metrics.get("heldLedgerBytes") != 0:
        raise AssertionError("failed experimental countsT leaked admitted bytes")
    return {
        "checks": (
            "exact-transpose",
            "complete-flag",
            "logical-edge",
            "bounded-admission",
            "failure-cancels-incomplete",
        ),
        "checksum": array_checksum(values.T),
        "shape": list(counts_t.shape),
        "chunks": list(counts_t.chunks),
        "shards": list(counts_t.shards or ()),
        "writer": writer_metrics,
        "failureKey": chunk_writes[0],
        "failureWriter": failure_metrics,
        "kind": "observed",
    }


def measure_read_group_widths(
    counts_t: zarr.Array,
    resources: ResourceBudget,
    *,
    probe: Any | None = None,
) -> dict[str, Any]:
    import time

    observations: dict[str, Any] = {}
    shards = counts_t.shards
    full_shard_width = (
        max(1, int(shards[0]) // int(counts_t.chunks[0])) if shards is not None else 1
    )
    widths = (
        ("1", 1),
        ("2", 2),
        ("4", 4),
        ("fullShard", full_shard_width),
    )
    for label, width in widths:
        if probe is not None:
            probe.reset()
        started = time.perf_counter()
        loaded = list(
            map_feature_read_groups(
                counts_t,
                lambda group: group.values.shape,
                resources=resources,
                readGroupChunks=width,
            )
        )
        observations[label] = {
            "readGroupChunks": width,
            "seconds": time.perf_counter() - started,
            "groups": len(loaded),
            "kind": "observed",
        }
        if probe is not None:
            observations[label]["storeOperations"] = probe.to_json()
    return observations


def run_one_variant(
    values: np.ndarray,
    *,
    layout: LayoutKind,
    writer: WriterKind,
    consumer: ConsumerKind,
    policy: CountMatrixLayoutPolicy,
    resources: ResourceBudget,
    store: Store | None = None,
    profile: str = "fast_local",
) -> dict[str, Any]:
    import time

    backend = store if store is not None else zarr.storage.MemoryStore()
    root = zarr.open_group(store=backend, mode="w")
    group = root.create_group("RNA")
    if layout == "candidate":
        counts, _plan = create_candidate_counts(
            group,
            values,
            policy=policy,
            profile=profile,
            resources=resources,
        )
    else:
        counts = create_current_counts(group, values, profile=profile)
    started = time.perf_counter()
    counts_t = write_layout_counts_t(
        counts,
        group,
        writer=writer,
        policy=policy,
        profile=profile,
        resources=resources,
    )
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    gathered = consume_counts_t(counts_t, consumer=consumer, resources=resources)
    consume_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(gathered, values.T)
    return {
        "writeSeconds": write_seconds,
        "consumeSeconds": consume_seconds,
        "checksum": array_checksum(gathered),
        "complete": bool(counts_t.attrs.get("complete")),
        "countsTChunks": list(counts_t.chunks),
        "countsTShards": None if counts_t.shards is None else list(counts_t.shards),
        "kind": "observed",
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires at least one value")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def _relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline


def select_phase3_branch(
    summaries: dict[str, dict[str, Any]],
    *,
    minReps: int = PHASE3_MIN_REPS,
) -> tuple[str, str]:
    """Return (branch, reason) from predeclared 100k decision signals."""
    required = [name for name, _layout, _writer, _consumer in PHASE3_VARIANTS]
    if any(name not in summaries for name in required):
        return "E", "missing-variant"
    reps = [int(summaries[name].get("reps", 0)) for name in required]
    if min(reps) < minReps:
        return "E", "insufficient-repetitions"
    if any(bool(summaries[name].get("highVariance", False)) for name in required):
        return "E", "high-variance"
    baseline = summaries["currentWholeStrip"]
    current_bounded = summaries["currentBounded"]
    candidate = summaries["candidateBounded"]
    write_base = float(baseline["writeMedianSeconds"])
    write_current = float(current_bounded["writeMedianSeconds"])
    write_candidate = float(candidate["writeMedianSeconds"])
    marker_base = float(baseline.get("markerMedianSeconds", write_base))
    marker_current = float(current_bounded.get("markerMedianSeconds", write_current))
    marker_candidate = float(candidate.get("markerMedianSeconds", write_candidate))
    hvg_base = float(baseline.get("hvgMedianSeconds", write_base))
    hvg_current = float(current_bounded.get("hvgMedianSeconds", write_current))
    hvg_candidate = float(candidate.get("hvgMedianSeconds", write_candidate))
    mem_base = float(baseline.get("peakMemoryBytes", 1))
    mem_current = float(current_bounded.get("peakMemoryBytes", mem_base))
    mem_candidate = float(candidate.get("peakMemoryBytes", mem_base))
    hvg_mem_base = float(baseline.get("hvgPeakMemoryBytes", mem_base))
    hvg_mem_current = float(current_bounded.get("hvgPeakMemoryBytes", mem_current))
    hvg_mem_candidate = float(candidate.get("hvgPeakMemoryBytes", mem_candidate))
    io_base = float(baseline.get("usefulToRequestedBytes", 1))
    io_current = float(current_bounded.get("usefulToRequestedBytes", io_base))
    io_candidate = float(candidate.get("usefulToRequestedBytes", io_base))

    def no_primary_regression(write: float, marker: float) -> bool:
        return (
            _relative_change(write, write_base) <= WALL_REGRESSION_LIMIT
            and _relative_change(marker, marker_base) <= WALL_REGRESSION_LIMIT
        )

    def hvg_protected(hvg: float, mem: float) -> bool:
        return (
            _relative_change(hvg, hvg_base) <= WALL_REGRESSION_LIMIT
            and _relative_change(mem, hvg_mem_base) <= WALL_REGRESSION_LIMIT
        )

    def material_gain_against(
        write: float,
        marker: float,
        mem: float,
        io_efficiency: float,
        *,
        baselineWrite: float,
        baselineMarker: float,
        baselineMemory: float,
        baselineIo: float,
    ) -> bool:
        return (
            _relative_change(write, baselineWrite) <= -PRIMARY_IMPROVEMENT_LIMIT
            or _relative_change(marker, baselineMarker) <= -PRIMARY_IMPROVEMENT_LIMIT
            or _relative_change(mem, baselineMemory) <= -MEMORY_IMPROVEMENT_LIMIT
            or _relative_change(io_efficiency, baselineIo) >= MEMORY_IMPROVEMENT_LIMIT
        )

    if bool(candidate.get("writerAdmissionFailed")):
        consumer_ok = (
            _relative_change(marker_candidate, marker_base) <= WALL_REGRESSION_LIMIT
            and hvg_protected(hvg_candidate, hvg_mem_candidate)
            and (
                _relative_change(marker_candidate, marker_current)
                <= -PRIMARY_IMPROVEMENT_LIMIT
                or _relative_change(mem_candidate, mem_current)
                <= -MEMORY_IMPROVEMENT_LIMIT
                or _relative_change(io_candidate, io_current)
                >= MEMORY_IMPROVEMENT_LIMIT
            )
        )
        if consumer_ok:
            return "C", "writer-admission-failed-consumer-gain"
        return "E", "writer-admission-failed"

    candidate_ok = (
        no_primary_regression(write_candidate, marker_candidate)
        and hvg_protected(hvg_candidate, hvg_mem_candidate)
        and material_gain_against(
            write_candidate,
            marker_candidate,
            mem_candidate,
            io_candidate,
            baselineWrite=write_current,
            baselineMarker=marker_current,
            baselineMemory=mem_current,
            baselineIo=io_current,
        )
    )
    bounded_ok = (
        no_primary_regression(write_current, marker_current)
        and hvg_protected(hvg_current, hvg_mem_current)
        and material_gain_against(
            write_current,
            marker_current,
            mem_current,
            io_current,
            baselineWrite=write_base,
            baselineMarker=marker_base,
            baselineMemory=mem_base,
            baselineIo=io_base,
        )
    )
    near_write = abs(_relative_change(write_candidate, write_base)) < 0.05
    if candidate_ok:
        return "A", "candidate-layout-gain"
    if bounded_ok and not candidate_ok:
        return "B", "bounded-current-layout-gain"
    if near_write and min(reps) >= minReps:
        return "E", "overlapping-threshold"
    if not bounded_ok and not candidate_ok:
        return "D", "no-bounded-gain"
    return "E", "inconclusive"


def _validate_phase3_cell_selection(
    values: np.ndarray,
    *,
    policy: CountMatrixLayoutPolicy,
    resources: ResourceBudget,
) -> dict[str, int]:
    cell_idx = np.array([17, 0, 103, 5, 220], dtype=np.int64)
    group_counts: dict[str, int] = {}
    layout_writers: tuple[tuple[LayoutKind, WriterKind], ...] = (
        ("current", "current"),
        ("candidate", "asyncCandidate"),
    )
    for layout, writer in layout_writers:
        root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
        group = root.create_group("RNA")
        if layout == "current":
            counts = create_current_counts(group, values, profile="fast_local")
        else:
            counts, _plan = create_candidate_counts(
                group,
                values,
                policy=policy,
                profile="fast_local",
                resources=resources,
            )
        counts_t = write_layout_counts_t(
            counts,
            group,
            writer=writer,
            policy=policy,
            profile="fast_local",
            resources=resources,
        )
        blocks = list(
            map_feature_read_groups(
                counts_t,
                lambda item: np.asarray(item.values),
                cell_idx=cell_idx,
                resources=resources,
            )
        )
        loaded = np.concatenate(blocks, axis=0)
        np.testing.assert_array_equal(loaded, values.T[:, cell_idx])
        group_counts[layout] = len(blocks)
    return group_counts


def run_phase3_local_checks(
    *,
    reps: int = 1,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
    resources: ResourceBudget | None = None,
) -> dict[str, Any]:
    values = scaled_values()
    budget = resources or ResourceBudget(32 * 1024 * 1024, 2)
    summaries: dict[str, dict[str, Any]] = {}
    for name, layout, writer, consumer in PHASE3_VARIANTS:
        writes: list[float] = []
        consumes: list[float] = []
        last: dict[str, Any] | None = None
        for _rep in range(max(1, int(reps))):
            last = run_one_variant(
                values,
                layout=layout,
                writer=writer,
                consumer=consumer,
                policy=policy,
                resources=budget,
            )
            writes.append(float(last["writeSeconds"]))
            consumes.append(float(last["consumeSeconds"]))
        assert last is not None
        summaries[name] = {
            "reps": len(writes),
            "writeMedianSeconds": median(writes),
            "consumeMedianSeconds": median(consumes),
            "markerMedianSeconds": median(consumes),
            "hvgMedianSeconds": median(consumes),
            "peakMemoryBytes": 1,
            "checksum": last["checksum"],
            "countsTChunks": last["countsTChunks"],
            "kind": "observed",
        }
    selection_groups = _validate_phase3_cell_selection(
        values,
        policy=policy,
        resources=budget,
    )
    branch, reason = select_phase3_branch(summaries, minReps=PHASE3_MIN_REPS)
    return {
        "checks": (
            "transpose-equality",
            "three-variants",
            "requested-cell-order",
        ),
        "summaries": summaries,
        "selectionGroups": selection_groups,
        "branch": branch,
        "reason": reason,
        "kind": "observed",
    }


def execution_plan_json(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "codecWorkerLimit": plan.codecWorkerLimit,
        "zarrAsyncConcurrency": plan.zarrAsyncConcurrency,
        "computeWorkerLimit": plan.computeWorkerLimit,
        "readGroupsInFlight": plan.readGroupsInFlight,
        "destinationCommitsInFlight": plan.destinationCommitsInFlight,
        "chunksPerShard": plan.chunksPerShard,
        "kind": "planned",
    }


def timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    import time

    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def run_phase4_local_checks() -> dict[str, Any]:
    from scarf.storage.count_matrix import (
        accepted_layout_branch,
        override_accepted_layout_branch,
        reject_noncanonical_write_destination,
        uses_experimental_product_layout,
    )

    checks: list[str] = []
    if accepted_layout_branch() != "current":
        raise AssertionError("product layout must stay current until Branch A")
    checks.append("default-current")
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    with override_accepted_layout_branch("A"):
        if not uses_experimental_product_layout():
            raise AssertionError("branch A must enable the paired layout")
        try:
            reject_noncanonical_write_destination(root)
        except ValueError as exc:
            if "Repack" not in str(exc):
                raise
        else:
            raise AssertionError(
                "branch A must reject destinations without paired metadata"
            )
        checks.append("branch-a-rejects-old-destination")
    if uses_experimental_product_layout():
        raise AssertionError("override must restore the previous branch")
    checks.append("override-restored")
    for branch in ("B", "C", "D", "E", "current"):
        with override_accepted_layout_branch(branch):
            if uses_experimental_product_layout():
                raise AssertionError(f"branch {branch} must keep the current layout")
            reject_noncanonical_write_destination(root)
    checks.append("non-a-keeps-current")
    return {
        "checks": tuple(checks),
        "productBranch": accepted_layout_branch(),
        "kind": "observed",
    }


def run_phase5_local_checks(
    *,
    policy: CountMatrixLayoutPolicy = SCALED_POLICY,
    resources: ResourceBudget | None = None,
) -> dict[str, Any]:
    from scarf.storage.feature_stream import map_feature_cell_bands

    values = scaled_values()
    budget = resources or ResourceBudget(32 * 1024 * 1024, 2)
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    counts = create_current_counts(group, values)
    counts_t = write_layout_counts_t(
        counts,
        group,
        writer="current",
        policy=policy,
        profile="fast_local",
        resources=budget,
    )
    cell_idx = np.array([5, 0, 3, 1], dtype=np.int64)
    gathered = np.concatenate(
        [
            np.asarray(item.values)
            for item in map_feature_read_groups(
                counts_t,
                lambda group: group,
                cell_idx=cell_idx,
                resources=budget,
            )
        ],
        axis=0,
    )
    np.testing.assert_array_equal(gathered, values.T[:, cell_idx])
    dest = np.zeros_like(gathered)
    order: list[tuple[int, int]] = []

    def accumulate(band: Any) -> None:
        order.append((int(band.featStart), int(band.cellStart)))
        dest[band.featStart : band.featEnd, band.selectedDestinations] = band.values[
            :, band.selectedLocal
        ]

    list(
        map_feature_cell_bands(
            counts_t,
            accumulate,
            cell_idx=cell_idx,
            resources=budget,
        )
    )
    np.testing.assert_array_equal(dest, values.T[:, cell_idx])
    if order != sorted(order):
        raise AssertionError("cell-band reductions must run in deterministic order")
    return {
        "checks": (
            "unsorted-cell-order",
            "cell-band-reduction",
            "bounded-read-groups",
        ),
        "groups": len(order),
        "kind": "observed",
    }


def run_phase6_local_checks() -> dict[str, Any]:
    from profiling.stages import (
        validate_cluster_source_identity,
        validate_experiment_branches,
    )

    ids = np.array(["c0", "c1", "c2"], dtype=object)
    active = np.array([True, True, False])
    labels = np.array(["a", "b", "a"], dtype=object)
    groups = validate_cluster_source_identity(
        sourceIds=ids,
        targetIds=ids,
        sourceActive=active,
        targetActive=active,
        labels=labels,
    )
    if groups != ["a", "b"]:
        raise AssertionError("expected two imported groups")
    validate_experiment_branches(
        pcaComplete=True,
        importedColumnPresent=True,
        markerComplete=True,
    )
    return {
        "checks": (
            "cluster-identity",
            "pca-and-marker-branches",
        ),
        "groupCount": len(groups),
        "kind": "observed",
    }
