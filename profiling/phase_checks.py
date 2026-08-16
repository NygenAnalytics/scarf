"""Durable phase-check protocol and Phase 0 hypothesis gates."""

import asyncio
import hashlib
import io
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, Field
from zarr.abc.store import Store

from profiling.config import ClusterSourceRef, ProfilingConfig
from profiling.provenance import provenance_from_config
from profiling.r2 import (
    get_bytes,
    get_json,
    get_text,
    join_uri,
    list_common_prefixes,
    object_exists,
    object_metadata,
    put_json_if_absent,
    put_bytes_if_absent,
    put_text_if_absent,
)
from profiling.recording_store import (
    RecordingMemoryStore,
    StoreProbe,
    wrap_recording_store,
)

type PhaseName = Literal[
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
]
type HypothesisId = Literal["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]
type PhaseDecision = Literal["accept", "reject", "blocked"]
type PhaseStatus = Literal["ok", "error"]

HYPOTHESIS_BY_PHASE: dict[PhaseName, HypothesisId] = {
    "phase0": "H0",
    "phase1": "H1",
    "phase2": "H2",
    "phase3": "H3",
    "phase4": "H8",
    "phase5": "H5",
    "phase6": "H7",
    "phase7": "H4",
    "phase8": "H4",
}

PHASE0_SYNTHETIC_CELLS = 221
PHASE0_SYNTHETIC_FEATS = 117
PHASE0_COUNTS_SHARDS = (50, 200)
PHASE0_COUNTS_CHUNKS = (50, 20)
PHASE0_COUNTST_SHARDS = (20, 100)
PHASE0_COUNTST_CHUNKS = (10, 50)
PHASE0_INVENTORY_SIZES = (10_000, 100_000, 1_000_000)
PHASE0_CLUSTER_LIST_LIMIT = 64
PHASE3_HVG_RTOL = 1e-10
PHASE3_HVG_ATOL = 1e-12
PHASE3_HVG_CUTOFF_TIE_BAND = 1e-10
PHASE3_PCA_FLOAT32_RTOL = 1e-6
PHASE3_PCA_FLOAT32_ATOL = 5e-7
PHASE3_PCA_FLOAT64_RTOL = 1e-10
PHASE3_PCA_FLOAT64_ATOL = 1e-12
CLUSTER_SOURCES_PATH = Path(__file__).resolve().parent / "cluster_sources.toml"


class PhaseClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: PhaseName
    hypothesis: HypothesisId
    runTag: str
    claimedAt: str
    sourceTreeSha256: str | None = None
    lockfileSha256: str | None = None
    configSha256: str | None = None


class PhaseWorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: PhaseName
    hypothesis: HypothesisId
    status: PhaseStatus
    checks: tuple[str, ...]
    deferredChecks: tuple[str, ...] = ()
    observations: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    provenance: dict[str, Any] | None = None


class PhaseReopenResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: PhaseName
    hypothesis: HypothesisId
    status: PhaseStatus
    validated: bool
    checks: tuple[str, ...]
    observations: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    provenance: dict[str, Any] | None = None


type Phase3VariantName = Literal[
    "currentWholeStrip",
    "currentBounded",
    "candidateBounded",
]
type ComparisonNamespace = Literal["phase3", "scale"]


class Phase3VariantResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repetition: int
    variant: Phase3VariantName
    runTag: str
    storeUri: str
    status: PhaseStatus
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    setup: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    provenance: dict[str, Any] | None = None


class ScaleComparisonClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runTag: str
    nRows: int
    repetitions: int
    pilotRepetitions: int
    evidenceRunTag: str
    baselineRunTag: str
    decisionRule: Literal["scale-focused"]
    claimedAt: str
    sourceTreeSha256: str | None = None
    lockfileSha256: str | None = None
    configSha256: str | None = None


class ScaleBatchValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nRows: int
    batch: Literal["pilot", "continuation"]
    repetitionStart: int
    repetitionEnd: int
    status: PhaseStatus
    validated: bool
    checks: tuple[str, ...]
    validations: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    provenance: dict[str, Any] | None = None


class ScaleComparisonFinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nRows: int
    status: PhaseStatus
    conclusion: Literal["measurement-complete", "measurement-failed"]
    decisionRule: Literal["scale-focused"]
    repetitions: int
    completedChecks: tuple[str, ...]
    summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    baseline100kSummaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scalingContext: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pilotValidationUri: str
    continuationValidationUri: str
    requiresReview: bool = True
    error: str | None = None
    provenance: dict[str, Any] | None = None


class PhaseFinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: PhaseName
    hypothesis: HypothesisId
    status: PhaseStatus
    decision: PhaseDecision
    branch: str
    nextPhase: PhaseName | None
    completedChecks: tuple[str, ...]
    deferredChecks: tuple[str, ...] = ()
    worker: PhaseWorkerResult
    reopen: PhaseReopenResult
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_create_only(uri: str, payload: dict[str, Any]) -> None:
    if not put_json_if_absent(uri, payload):
        raise RuntimeError(f"Refusing to overwrite existing object at {uri}")


def conflicting_phase_uris(config: ProfilingConfig, phase: PhaseName) -> list[str]:
    candidates = [
        config.phaseWorkerResultUri(phase),
        config.phaseReopenResultUri(phase),
        config.phaseFinalResultUri(phase),
        f"{config.phaseSyntheticStoreUri(phase).rstrip('/')}/zarr.json",
    ]
    if phase == "phase0":
        candidates.extend(
            [
                *(
                    config.phaseInputManifestUri("phase0", n_rows)
                    for n_rows in PHASE0_INVENTORY_SIZES
                ),
                config.phaseClusterInventoryUri("phase0"),
                config.phaseClusterSourcesTomlUri("phase0"),
            ]
        )
    if phase == "phase2":
        candidates.append(f"{config.phaseFailureStoreUri(phase).rstrip('/')}/zarr.json")
        candidates.extend(
            f"{config.phaseSweepStoreUri(phase, readGroupsInFlight=reads, destinationCommitsInFlight=commits).rstrip('/')}/zarr.json"
            for reads, commits in ((2, 1), (1, 2))
        )
    if phase == "phase3":
        candidates.extend(
            [
                config.phase3ScheduleUri(),
                config.phase3ReferenceUri(),
                config.phase3ReferenceArraysUri(),
                *(
                    config.phase3ValidationUri(repetition, variant)
                    for repetition in range(3)
                    for variant in (
                        "currentWholeStrip",
                        "currentBounded",
                        "candidateBounded",
                    )
                ),
            ]
        )
    return [uri for uri in candidates if object_exists(uri)]


def claim_phase(config: ProfilingConfig, phase: PhaseName) -> PhaseClaim:
    if not config.runTag.strip():
        raise ValueError("phase-check requires a non-empty runTag")
    conflicts = conflicting_phase_uris(config, phase)
    if conflicts:
        raise RuntimeError(
            "Refusing conflicting phase destinations: " + ", ".join(conflicts)
        )
    provenance = provenance_from_config(config)
    claim = PhaseClaim(
        phase=phase,
        hypothesis=HYPOTHESIS_BY_PHASE[phase],
        runTag=config.runTag,
        claimedAt=utc_now(),
        sourceTreeSha256=provenance.get("sourceTreeSha256"),
        lockfileSha256=provenance.get("lockfileSha256"),
        configSha256=provenance.get("configSha256"),
    )
    if not put_json_if_absent(
        config.phaseClaimUri(phase), claim.model_dump(mode="json")
    ):
        raise RuntimeError(f"Duplicate phase claim at {config.phaseClaimUri(phase)}")
    return claim


def resume_phase_claim(config: ProfilingConfig, phase: PhaseName) -> PhaseClaim:
    """Load a durable claim for explicit recovery of an interrupted coordinator."""
    claim = PhaseClaim.model_validate(get_json(config.phaseClaimUri(phase)))
    if claim.phase != phase or claim.hypothesis != HYPOTHESIS_BY_PHASE[phase]:
        raise RuntimeError(f"Existing claim does not match {phase}")
    if claim.runTag != config.runTag:
        raise RuntimeError("Existing claim runTag does not match the configuration")
    current = provenance_from_config(config)
    if claim.configSha256 != current.get("configSha256"):
        raise RuntimeError("Existing claim config fingerprint does not match")
    if claim.lockfileSha256 != current.get("lockfileSha256"):
        raise RuntimeError("Existing claim lockfile fingerprint does not match")
    return claim


def write_worker_result(
    config: ProfilingConfig,
    result: PhaseWorkerResult,
) -> str:
    uri = config.phaseWorkerResultUri(result.phase)
    _write_create_only(uri, result.model_dump(mode="json"))
    return uri


def write_reopen_result(
    config: ProfilingConfig,
    result: PhaseReopenResult,
) -> str:
    uri = config.phaseReopenResultUri(result.phase)
    _write_create_only(uri, result.model_dump(mode="json"))
    return uri


def write_final_result(
    config: ProfilingConfig,
    result: PhaseFinalResult,
) -> str:
    uri = config.phaseFinalResultUri(result.phase)
    _write_create_only(uri, result.model_dump(mode="json"))
    return uri


def synthetic_count_values(
    nCells: int = PHASE0_SYNTHETIC_CELLS,
    nFeats: int = PHASE0_SYNTHETIC_FEATS,
    *,
    dtype: str = "uint16",
) -> np.ndarray:
    cells = np.arange(nCells, dtype=np.uint32)[:, None]
    feats = np.arange(nFeats, dtype=np.uint32)[None, :]
    encoded = cells * np.uint32(nFeats) + feats
    info = np.iinfo(np.dtype(dtype))
    return (encoded % np.uint32(info.max)).astype(dtype)


def fill_synthetic_pair(store: Store) -> tuple[zarr.Array, zarr.Array]:
    root = zarr.open_group(store=store, mode="w")
    values = synthetic_count_values()
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=PHASE0_COUNTS_CHUNKS,
        shards=PHASE0_COUNTS_SHARDS,
        dtype=values.dtype,
        overwrite=True,
    )
    counts[:] = values
    counts_t = root.create_array(
        "countsT",
        shape=(values.shape[1], values.shape[0]),
        chunks=PHASE0_COUNTST_CHUNKS,
        shards=PHASE0_COUNTST_SHARDS,
        dtype=values.dtype,
        overwrite=True,
    )
    counts_t[:] = values.T
    return counts, counts_t


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_zarr_fact_checks(counts: zarr.Array, countsT: zarr.Array) -> list[str]:
    values = synthetic_count_values()
    _assert(counts.shards == PHASE0_COUNTS_SHARDS, "counts shards mismatch")
    _assert(counts.chunks == PHASE0_COUNTS_CHUNKS, "counts chunks mismatch")
    block00 = np.asarray(counts.blocks[0, 0])
    _assert(
        block00.shape == (50, 117), f"outer block clipped width, got {block00.shape}"
    )
    omitted = np.asarray(counts.blocks[0])
    _assert(
        omitted.shape == (50, 117),
        f"omitted block axis should span remaining shards, got {omitted.shape}",
    )
    last = np.asarray(counts.blocks[4, 0])
    _assert(last.shape == (21, 117), f"last outer block should clip, got {last.shape}")
    inner = np.asarray(counts[0:50, 0:20])
    _assert(inner.shape == (50, 20), "inner-chunk slice shape")
    np.testing.assert_array_equal(inner, values[0:50, 0:20])
    edge = np.asarray(counts[200:221, 100:117])
    _assert(edge.shape == (21, 17), "logical edge slice shape")
    np.testing.assert_array_equal(edge, values[200:221, 100:117])
    _assert(countsT.blocks[0, 0].shape == (20, 100), "countsT outer shard")
    _assert(countsT.blocks[0].shape == (20, 221), "countsT omitted cell axis")
    _assert(countsT.blocks[5, 2].shape == (17, 21), "countsT clipped last shard")
    np.testing.assert_array_equal(np.asarray(countsT[:]), values.T)

    async def _roundtrip() -> None:
        got = np.asarray(
            await counts.async_array.getitem((slice(200, 221), slice(100, 117)))
        )
        np.testing.assert_array_equal(got, values[200:221, 100:117])
        await countsT.async_array.setitem(
            (slice(0, 1), slice(0, 1)),
            np.asarray(countsT[0:1, 0:1]),
        )
        restored = np.asarray(
            await countsT.async_array.getitem((slice(0, 1), slice(0, 1)))
        )
        np.testing.assert_array_equal(restored, values.T[0:1, 0:1])

    asyncio.run(_roundtrip())
    return [
        "outer-block-index",
        "omitted-block-axis",
        "logical-edge-clip",
        "inner-chunk-slice",
        "async-roundtrip",
        "transpose-values",
    ]


def measure_store_read_groups(counts: zarr.Array, probe: StoreProbe) -> dict[str, Any]:
    n_cells, n_feats = (int(value) for value in counts.shape)
    chunk_c, chunk_f = (int(value) for value in counts.chunks)
    observations: dict[str, Any] = {}
    selections = {
        "oneInnerChunk": (slice(0, chunk_c), slice(0, chunk_f)),
        "twoAdjacentChunks": (slice(0, chunk_c), slice(0, min(n_feats, chunk_f * 2))),
        "fourAdjacentChunks": (slice(0, chunk_c), slice(0, min(n_feats, chunk_f * 4))),
        "completeShard": (
            slice(0, min(n_cells, int(counts.shards[0]))),
            slice(0, n_feats),
        ),
    }
    for name, selection in selections.items():
        probe.reset()
        started = time.perf_counter()
        payload = np.asarray(counts[selection])
        elapsed = time.perf_counter() - started
        observations[name] = {
            "shape": list(payload.shape),
            "seconds": elapsed,
            "store": probe.to_json(),
            "kind": "observed",
        }
    return observations


def run_current_path_baselines(values: np.ndarray) -> dict[str, Any]:
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.feature_stream import load_feature_strip, plan_feature_stream
    from scarf.storage.layout import count_array_spec, iter_shard_row_slices
    from scarf.storage.sharding import write_counts_t

    store = RecordingMemoryStore()
    root = zarr.open_group(store=store, mode="w")
    group = root.create_group("RNA")
    spec = count_array_spec(
        values.shape[0],
        values.shape[1],
        dtype=values.dtype,
        profile="fast_local",
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
    resources = ResourceBudget(512 * 1024 * 1024, 2)
    store.reset()
    started = time.perf_counter()
    counts_t = write_counts_t(counts, group, resources=resources)
    write_seconds = time.perf_counter() - started
    if counts_t is None:
        raise AssertionError("current countsT writer returned None")
    np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
    writer_ops = store.probe.to_json()
    store.reset()
    started = time.perf_counter()
    shard = load_feature_strip(counts_t, 0)
    strip_seconds = time.perf_counter() - started
    _assert(shard.values.shape[1] == values.shape[0], "whole-strip loads all cells")
    plan = plan_feature_stream(
        counts_t,
        featureAxis=0,
        cellAxis=1,
        featureIndices=np.arange(min(8, values.shape[1])),
        cellIndices=np.arange(values.shape[0]),
        resources=resources,
        blockBytes=lambda width: int(width) * values.shape[0] * values.dtype.itemsize,
    )
    row_slices = list(iter_shard_row_slices(values.shape[0], int(counts.chunks[0])))
    return {
        "kind": "observed",
        "writeCountsTSeconds": write_seconds,
        "writeCountsTStore": writer_ops,
        "wholeStripSeconds": strip_seconds,
        "wholeStripShape": list(shard.values.shape),
        "featureStreamBlocks": len(plan.blocks),
        "rowStreamSlices": len(row_slices),
        "countsChunks": list(counts.chunks),
        "countsShards": None if counts.shards is None else list(counts.shards),
        "countsTChunks": list(counts_t.chunks),
        "countsTShards": None if counts_t.shards is None else list(counts_t.shards),
    }


def _ordered_id_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"scarf-ordered-ids-v1\0")
    digest.update(np.int64(values.shape[0]).tobytes())
    for item in values:
        if isinstance(item, bytes | bytearray | np.bytes_):
            payload = bytes(item)
        else:
            payload = str(item).encode()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def inspect_h5ad_manifest(
    path: Path, *, objectMeta: dict[str, Any] | None
) -> dict[str, Any]:
    from scarf.readers import H5adReader

    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            sha256.update(chunk)
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        storage_dtype = np.dtype(reader.infer_storage_dtype()).name
        cell_ids = np.asarray(reader.cell_ids())
        feature_ids = np.asarray(reader.feat_ids())
        nnz = int(getattr(reader, "nCounts", 0) or 0)
        if nnz == 0:
            matrix = reader.h5[reader.matrixKey]
            if hasattr(matrix, "keys") and "data" in matrix:
                nnz = int(matrix["data"].shape[0])
        return {
            "kind": "observed",
            "sha256": sha256.hexdigest(),
            "fileBytes": int(path.stat().st_size),
            "eTag": None if objectMeta is None else objectMeta.get("eTag"),
            "nRows": int(reader.nCells),
            "nColumns": int(reader.nFeatures),
            "nnz": nnz,
            "orderedRowDigest": _ordered_id_digest(cell_ids),
            "orderedFeatureDigest": _ordered_id_digest(feature_ids),
            "inferredStorageDtype": storage_dtype,
            "sourceMatrixDtype": str(np.dtype(reader.matrixDtype).name),
        }
    finally:
        reader.h5.close()


def inventory_h5ad_objects(
    config: ProfilingConfig,
    workDir: Path,
    *,
    sizes: tuple[int, ...] = PHASE0_INVENTORY_SIZES,
) -> dict[str, Any]:
    from profiling.datasets import SOURCE_SPEC
    from profiling.r2 import download_file

    manifests: dict[str, Any] = {}
    workDir.mkdir(parents=True, exist_ok=True)
    for n_rows in sizes:
        uri = config.datasetUri(n_rows)
        meta = object_metadata(uri)
        record: dict[str, Any] = {
            "uri": uri,
            "kind": "observed",
            "present": meta is not None,
            "objectMeta": meta,
            "expectedRows": n_rows,
            "sourceDatasetId": SOURCE_SPEC.datasetId,
            "sourceVersionId": SOURCE_SPEC.versionId,
            "samplingSeed": config.samplingSeed,
        }
        if meta is None:
            record["status"] = "missing"
            manifests[str(n_rows)] = record
            continue
        local = workDir / f"{n_rows}.h5ad"
        download_file(uri, local)
        record.update(inspect_h5ad_manifest(local, objectMeta=meta))
        record["status"] = "ok"
        manifests[str(n_rows)] = record
    return manifests


def _read_string_array(array: zarr.Array) -> np.ndarray:
    return np.asarray(array[:])


def inspect_cluster_source(
    storeUri: str,
    *,
    expectedRows: int,
    labelColumn: str,
    storageOptions: dict[str, str] | None,
) -> dict[str, Any]:
    from scarf.storage.stores import open_store

    root = open_store(storeUri, mode="r", storage_options=storageOptions)
    if "cellData" not in root:
        return {
            "uri": storeUri,
            "status": "missing-cellData",
            "kind": "observed",
        }
    cell_data = root["cellData"]
    columns = sorted(str(name) for name in cell_data.keys())
    if "ids" not in cell_data:
        return {
            "uri": storeUri,
            "status": "missing-ids",
            "columns": columns,
            "kind": "observed",
        }
    ids = _read_string_array(cell_data["ids"])
    n_rows = int(ids.shape[0])
    unique_ids = int(np.unique(ids).shape[0])
    active = None
    if "I" in cell_data:
        active = np.asarray(cell_data["I"][:]).astype(bool)
    labels = None
    source_artifact = None
    label_present = labelColumn in cell_data
    if label_present:
        label_array = cell_data[labelColumn]
        labels = _read_string_array(label_array)
        source_artifact = dict(label_array.attrs).get("source_artifact")
    feature_ids = None
    if "RNA" in root and "featureData" in root["RNA"]:
        feature_group = root["RNA"]["featureData"]
        if "ids" in feature_group:
            feature_ids = _read_string_array(feature_group["ids"])
    groups = []
    complete_labels = False
    if labels is not None and active is not None:
        active_labels = labels[active]
        complete_labels = all(
            str(value) not in {"", "nan", "None"} for value in active_labels
        )
        groups = sorted({str(value) for value in active_labels})
    status = "candidate"
    if n_rows != expectedRows:
        status = "row-count-mismatch"
    elif unique_ids != n_rows:
        status = "duplicate-ids"
    elif not label_present:
        status = "missing-label"
    elif active is None:
        status = "missing-active-mask"
    elif not complete_labels:
        status = "incomplete-labels"
    elif len(groups) < 2:
        status = "one-group"
    return {
        "uri": storeUri,
        "status": status,
        "kind": "observed",
        "nRows": n_rows,
        "uniqueCellIds": unique_ids,
        "orderedCellDigest": _ordered_id_digest(ids),
        "orderedFeatureDigest": None
        if feature_ids is None
        else _ordered_id_digest(feature_ids),
        "labelColumn": labelColumn,
        "labelPresent": label_present,
        "sourceArtifact": source_artifact,
        "activeCells": None if active is None else int(active.sum()),
        "groupCount": len(groups),
        "columns": columns,
        "complete": status == "candidate",
    }


def inventory_cluster_sources(
    config: ProfilingConfig,
    *,
    sizes: tuple[int, ...] = PHASE0_INVENTORY_SIZES,
) -> dict[str, Any]:
    from profiling.r2 import storage_options

    options = storage_options(config.resultsUri)
    prefix = join_uri(config.resultsUri, "stores")
    configured = {item.nRows: item for item in config.clusterSources}
    discovered: list[str] = []
    try:
        run_prefixes = list_common_prefixes(prefix)[:PHASE0_CLUSTER_LIST_LIMIT]
        for run_prefix in run_prefixes:
            remaining = PHASE0_CLUSTER_LIST_LIMIT - len(discovered)
            if remaining <= 0:
                break
            discovered.extend(list_common_prefixes(run_prefix)[:remaining])
    except Exception as exc:
        discovered = []
        list_error = f"{type(exc).__name__}: {exc}"
    else:
        list_error = None
    inventory: dict[str, Any] = {}
    for n_rows in sizes:
        explicit = configured.get(n_rows)
        candidates: list[str] = []
        if explicit is not None:
            candidates.append(explicit.storeUri)
        suffix = f"/{n_rows}.zarr"
        for uri in discovered:
            if uri.endswith(suffix):
                if uri not in candidates:
                    candidates.append(uri)
            if len(candidates) >= 4:
                break
        inspected = [
            inspect_cluster_source(
                uri,
                expectedRows=n_rows,
                labelColumn=explicit.labelColumn
                if explicit is not None
                else "RNA_leiden_cluster",
                storageOptions=options,
            )
            for uri in candidates[:4]
        ]
        valid = [item for item in inspected if item.get("status") == "candidate"]
        inventory[str(n_rows)] = {
            "kind": "observed",
            "h7": "accepted" if valid else "blocked",
            "candidates": inspected,
            "listError": list_error,
        }
    return inventory


def reconcile_cluster_inventory(
    inventory: dict[str, Any],
    manifests: dict[str, Any],
) -> dict[str, Any]:
    reconciled = json.loads(json.dumps(inventory))
    for size, payload in reconciled.items():
        manifest = manifests.get(size, {})
        expected_cells = manifest.get("orderedRowDigest")
        expected_features = manifest.get("orderedFeatureDigest")
        for candidate in payload.get("candidates", []):
            if candidate.get("status") != "candidate":
                continue
            if (
                candidate.get("orderedCellDigest") != expected_cells
                or candidate.get("orderedFeatureDigest") != expected_features
            ):
                candidate["status"] = "input-identity-mismatch"
                candidate["complete"] = False
        valid = [
            candidate
            for candidate in payload.get("candidates", [])
            if candidate.get("status") == "candidate"
        ]
        payload["h7"] = "accepted" if valid else "blocked"
    return reconciled


def render_cluster_sources_toml(inventory: dict[str, Any]) -> str:
    lines = [
        "# Generated by phase-check. Do not commit.",
        "# Explicit cluster-source URIs for imported marker workloads.",
        "",
    ]
    for n_rows, payload in inventory.items():
        valid = [
            item
            for item in payload.get("candidates", [])
            if item.get("status") == "candidate"
        ]
        if not valid:
            continue
        chosen = valid[0]
        lines.extend(
            [
                "[[clusterSources]]",
                f"nRows = {int(n_rows)}",
                f'storeUri = "{chosen["uri"]}"',
                f'labelColumn = "{chosen.get("labelColumn", "RNA_leiden_cluster")}"',
                "",
            ]
        )
    return "\n".join(lines)


def write_cluster_sources_file(
    inventory: dict[str, Any],
    path: Path = CLUSTER_SOURCES_PATH,
) -> Path | None:
    text = render_cluster_sources_toml(inventory)
    if "[[clusterSources]]" not in text:
        return None
    path.write_text(text)
    return path


def run_phase0_local_checks() -> dict[str, Any]:
    store = RecordingMemoryStore()
    counts, counts_t = fill_synthetic_pair(store)
    checks = run_zarr_fact_checks(counts, counts_t)
    read_groups = measure_store_read_groups(counts, store.probe)
    baselines = run_current_path_baselines(synthetic_count_values())
    return {
        "checks": checks,
        "readGroups": read_groups,
        "currentPathBaselines": baselines,
        "store": store.probe.to_json(),
    }


def run_phase0_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemoteInventory: bool,
) -> PhaseWorkerResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {
        "hypothesis": "H0",
        "kind": "observed",
        "remoteInventoryIncluded": includeRemoteInventory,
    }
    try:
        local = run_phase0_local_checks()
        checks.extend(local["checks"])
        observations["readGroups"] = local["readGroups"]
        observations["currentPathBaselines"] = local["currentPathBaselines"]
        checks.append("current-path-baselines")
        if includeRemoteInventory:
            from scarf.storage.stores import make_store

            from profiling.r2 import storage_options

            synthetic_uri = config.phaseSyntheticStoreUri("phase0")
            options = storage_options(synthetic_uri)
            remote_store = wrap_recording_store(
                make_store(synthetic_uri, storage_options=options)
            )
            counts, counts_t = fill_synthetic_pair(remote_store)
            checks.extend(
                f"r2-{name}" for name in run_zarr_fact_checks(counts, counts_t)
            )
            observations["r2ReadGroups"] = measure_store_read_groups(
                counts,
                remote_store.probe,
            )
            observations["h5adManifests"] = inventory_h5ad_objects(config, workDir)
            missing_inputs: list[str] = []
            for size, manifest in observations["h5adManifests"].items():
                if not put_json_if_absent(
                    config.phaseInputManifestUri("phase0", int(size)),
                    manifest,
                ):
                    raise FileExistsError(
                        f"phase0 input manifest already exists for {size}"
                    )
                if manifest.get("status") != "ok":
                    missing_inputs.append(size)
            checks.append("h5ad-inventory")
            checks.append("h5ad-inventory-durable")
            if missing_inputs:
                observations["inputPolicy"] = {
                    "status": "blocked",
                    "message": (
                        "missing required input artifacts for sizes "
                        + ", ".join(missing_inputs)
                    ),
                }
            observations["clusterSources"] = reconcile_cluster_inventory(
                inventory_cluster_sources(config),
                observations["h5adManifests"],
            )
            if not put_json_if_absent(
                config.phaseClusterInventoryUri("phase0"),
                observations["clusterSources"],
            ):
                raise FileExistsError("phase0 cluster inventory already exists")
            cluster_sources_text = render_cluster_sources_toml(
                observations["clusterSources"]
            )
            if not put_text_if_absent(
                config.phaseClusterSourcesTomlUri("phase0"),
                cluster_sources_text,
            ):
                raise FileExistsError("phase0 cluster source TOML already exists")
            checks.append("cluster-source-inventory")
            checks.append("cluster-source-inventory-durable")
            dtype_decisions = {}
            for size, manifest in observations["h5adManifests"].items():
                inferred = manifest.get("inferredStorageDtype")
                dtype_decisions[size] = {
                    "inferredStorageDtype": inferred,
                    "uint16": inferred == "uint16",
                }
                if manifest.get("status") == "ok" and inferred != "uint16":
                    observations["dtypePolicy"] = {
                        "status": "blocked",
                        "message": (
                            f"size {size} inferred {inferred}; do not cast silently"
                        ),
                    }
            observations["dtypeDecisions"] = dtype_decisions
        return PhaseWorkerResult(
            phase="phase0",
            hypothesis="H0",
            status="ok",
            checks=tuple(checks),
            deferredChecks=(),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseWorkerResult(
            phase="phase0",
            hypothesis="H0",
            status="error",
            checks=tuple(checks),
            deferredChecks=(),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase0_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase0"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase0 worker failed")
        checks.append("worker-result-present")
        synthetic_uri = config.phaseSyntheticStoreUri("phase0")
        if object_exists(f"{synthetic_uri.rstrip('/')}/zarr.json"):
            from scarf.storage.stores import open_store

            from profiling.r2 import storage_options

            root = open_store(
                synthetic_uri,
                mode="r",
                storage_options=storage_options(synthetic_uri),
            )
            values = synthetic_count_values()
            np.testing.assert_array_equal(np.asarray(root["counts"][:]), values)
            np.testing.assert_array_equal(np.asarray(root["countsT"][:]), values.T)
            checks.append("synthetic-reopen-values")
        if worker.observations.get("remoteInventoryIncluded", False):
            worker_manifests = worker.observations.get("h5adManifests", {})
            for size, expected in worker_manifests.items():
                actual = get_json(config.phaseInputManifestUri("phase0", int(size)))
                if actual != expected:
                    raise ValueError(
                        f"phase0 durable input manifest differs for {size}"
                    )
            checks.append("durable-input-manifests")
            durable_cluster_inventory = get_json(
                config.phaseClusterInventoryUri("phase0")
            )
            if durable_cluster_inventory != worker.observations.get(
                "clusterSources",
                {},
            ):
                raise ValueError("phase0 durable cluster inventory differs")
            durable_cluster_sources = get_text(
                config.phaseClusterSourcesTomlUri("phase0")
            )
            if durable_cluster_sources != render_cluster_sources_toml(
                durable_cluster_inventory
            ):
                raise ValueError("phase0 durable cluster source TOML differs")
            checks.append("durable-cluster-inventory")
        return PhaseReopenResult(
            phase="phase0",
            hypothesis="H0",
            status="ok",
            validated=True,
            checks=tuple(checks),
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseReopenResult(
            phase="phase0",
            hypothesis="H0",
            status="error",
            validated=False,
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase0(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    blocked_h7 = False
    dtype_blocked = False
    input_blocked = False
    if worker.status == "ok":
        cluster = worker.observations.get("clusterSources", {})
        blocked_h7 = any(payload.get("h7") == "blocked" for payload in cluster.values())
        dtype_blocked = (
            worker.observations.get("dtypePolicy", {}).get("status") == "blocked"
        )
        input_blocked = (
            worker.observations.get("inputPolicy", {}).get("status") == "blocked"
        )
    if worker.status != "ok" or reopen.status != "ok" or not reopen.validated:
        decision: PhaseDecision = "reject"
        next_phase = None
        branch = "stop-harness"
        error = worker.error or reopen.error
    elif input_blocked:
        decision = "blocked"
        next_phase = None
        branch = "input-inventory"
        error = worker.observations["inputPolicy"]["message"]
    elif dtype_blocked:
        decision = "blocked"
        next_phase = None
        branch = "dtype-policy"
        error = worker.observations["dtypePolicy"]["message"]
    else:
        decision = "accept"
        next_phase = "phase1"
        branch = "phase1"
        error = None
        if blocked_h7:
            branch = "phase1-h7-blocked"
    return PhaseFinalResult(
        phase="phase0",
        hypothesis="H0",
        status="ok" if decision == "accept" else worker.status,
        decision=decision,
        branch=branch,
        nextPhase=next_phase,
        completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
        deferredChecks=worker.deferredChecks,
        worker=worker,
        reopen=reopen,
        error=error,
    )


def dumps_result(result: BaseModel) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)


def _policy_from_config(config: ProfilingConfig) -> Any:
    from scarf.storage.count_matrix import EXPERIMENTAL_POLICY, CountMatrixLayoutPolicy

    if config.countMatrixLayout is None:
        return EXPERIMENTAL_POLICY
    layout = config.countMatrixLayout
    return CountMatrixLayoutPolicy(
        targetReadUnitBytes=layout.targetReadUnitBytes,
        targetChunkBytes=layout.targetChunkBytes,
    )


def run_phase1_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.lattice_experiment import (
        SCALED_POLICY,
        fill_phase1_remote_pair,
        run_phase1_local_checks,
        scaled_values,
    )

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H1", "kind": "observed"}
    try:
        local = run_phase1_local_checks()
        checks.extend(local["checks"])
        observations["local"] = local
        if includeRemote:
            from scarf.storage.stores import make_store

            from profiling.r2 import storage_options

            uri = config.phaseSyntheticStoreUri("phase1")
            remote = make_store(uri, storage_options=storage_options(uri))
            observations["remote"] = fill_phase1_remote_pair(
                remote,
                policy=SCALED_POLICY,
                canonicalPolicy=_policy_from_config(config),
            )
            checks.append("r2-scaled-pair")
            observations["canonicalValues"] = {
                "nCells": int(scaled_values().shape[0]),
                "nFeats": int(scaled_values().shape[1]),
                "kind": "derived",
            }
        return PhaseWorkerResult(
            phase="phase1",
            hypothesis="H1",
            status="ok",
            checks=tuple(checks),
            deferredChecks=(),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseWorkerResult(
            phase="phase1",
            hypothesis="H1",
            status="error",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase1_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase1"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase1 worker failed")
        checks.append("worker-result-present")
        uri = config.phaseSyntheticStoreUri("phase1")
        if object_exists(f"{uri.rstrip('/')}/zarr.json"):
            from scarf.storage.stores import open_store

            from profiling.lattice_experiment import scaled_values
            from profiling.r2 import storage_options

            root = open_store(uri, mode="r", storage_options=storage_options(uri))
            values = scaled_values()
            np.testing.assert_array_equal(np.asarray(root["RNA"]["counts"][:]), values)
            stored = dict(root.attrs.get("scarf:countMatrixLayout", {}))
            if stored.get("fingerprint") != worker.observations.get("remote", {}).get(
                "fingerprint"
            ):
                raise AssertionError("reopened fingerprint does not match worker")
            canonical = root["canonical10k"]
            canonical_counts = canonical["counts"]
            canonical_counts_t = canonical["countsT"]
            if (
                np.asarray(canonical_counts[0, 0]).item() != 17
                or np.asarray(canonical_counts[-1, -1]).item() != 23
                or np.asarray(canonical_counts_t[0, 0]).item() != 17
                or np.asarray(canonical_counts_t[-1, -1]).item() != 23
            ):
                raise AssertionError("canonical 10k boundary values changed")
            canonical_stored = dict(canonical.attrs.get("scarf:countMatrixLayout", {}))
            expected_canonical = worker.observations.get("remote", {}).get(
                "canonical10k",
                {},
            )
            if canonical_stored.get("fingerprint") != expected_canonical.get(
                "fingerprint"
            ):
                raise AssertionError("canonical 10k fingerprint does not match worker")
            checks.extend(("reopen-values-policy", "canonical-10k-boundaries"))
        return PhaseReopenResult(
            phase="phase1",
            hypothesis="H1",
            status="ok",
            validated=True,
            checks=tuple(checks),
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseReopenResult(
            phase="phase1",
            hypothesis="H1",
            status="error",
            validated=False,
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase1(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    if worker.status != "ok" or reopen.status != "ok" or not reopen.validated:
        return PhaseFinalResult(
            phase="phase1",
            hypothesis="H1",
            status="error",
            decision="reject",
            branch="stop-planner",
            nextPhase=None,
            completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
            deferredChecks=worker.deferredChecks,
            worker=worker,
            reopen=reopen,
            error=worker.error or reopen.error,
        )
    return PhaseFinalResult(
        phase="phase1",
        hypothesis="H1",
        status="ok",
        decision="accept",
        branch="phase2",
        nextPhase="phase2",
        completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
        deferredChecks=(),
        worker=worker,
        reopen=reopen,
    )


def run_phase2_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from scarf.storage.async_execution import (
        configure_zarr_runtime,
        reset_zarr_runtime_for_tests,
    )
    from scarf.storage.budget import ResourceBudget

    from profiling.lattice_experiment import (
        SCALED_POLICY,
        create_candidate_counts,
        measure_read_group_widths,
        run_phase2_local_checks,
        scaled_values,
        write_layout_counts_t,
    )

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H2", "kind": "observed"}
    try:
        reset_zarr_runtime_for_tests()
        execution = config.executionPolicy
        if execution is not None:
            configure_zarr_runtime(
                codecWorkers=execution.codecWorkerLimit,
                asyncConcurrency=execution.zarrAsyncConcurrency,
            )
        budget = ResourceBudget(
            config.resourcesFor("writeCountsT").scarfMemoryBudget,
            config.resourcesFor("writeCountsT").workers,
        )
        local = run_phase2_local_checks(resources=budget)
        checks.extend(local["checks"])
        observations["local"] = local
        if includeRemote:
            from scarf.storage.stores import make_store

            from profiling.r2 import storage_options

            uri = config.phaseSyntheticStoreUri("phase2")
            remote = wrap_recording_store(
                make_store(uri, storage_options=storage_options(uri))
            )
            root = zarr.open_group(store=remote, mode="w")
            group = root.create_group("RNA")
            values = scaled_values()
            counts, _plan = create_candidate_counts(
                group,
                values,
                policy=SCALED_POLICY,
                profile="cloud",
                resources=budget,
            )
            remote.probe.reset()
            writer_metrics: dict[str, Any] = {}
            counts_t = write_layout_counts_t(
                counts,
                group,
                writer="asyncCandidate",
                policy=SCALED_POLICY,
                profile="cloud",
                resources=budget,
                readGroupChunks=1 if execution is None else execution.readGroupChunks,
                readGroupsInFlight=1
                if execution is None
                else execution.readGroupsInFlight,
                destinationCommitsInFlight=1
                if execution is None
                else execution.destinationCommitsInFlight,
                metrics=writer_metrics,
            )
            writer_store = remote.probe.to_json()
            chunk_writes = [
                key
                for operation, key in remote.probe.ops
                if operation == "set" and "/countsT/c/" in f"/{key}"
            ]
            if not chunk_writes:
                raise AssertionError("Phase 2 R2 writer emitted no countsT shard write")
            remote.probe.reset()
            np.testing.assert_array_equal(np.asarray(counts_t[:]), values.T)
            if int(writer_metrics.get("sourceDecodeBytes", 0)) < int(
                writer_metrics.get("sourceLogicalBytes", 0)
            ):
                raise AssertionError(
                    "Phase 2 source decode accounting is below logical source bytes"
                )
            observations["remote"] = {
                "checksum": local["checksum"],
                "readGroupWidths": measure_read_group_widths(
                    counts_t,
                    budget,
                    probe=remote.probe,
                ),
                "complete": bool(counts_t.attrs.get("complete")),
                "writer": writer_metrics,
                "writerStore": writer_store,
                "kind": "observed",
            }
            sweep_observations: dict[str, Any] = {}
            for reads_in_flight, commits_in_flight in ((2, 1), (1, 2)):
                sweep_uri = config.phaseSweepStoreUri(
                    "phase2",
                    readGroupsInFlight=reads_in_flight,
                    destinationCommitsInFlight=commits_in_flight,
                )
                sweep_store = wrap_recording_store(
                    make_store(
                        sweep_uri,
                        storage_options=storage_options(sweep_uri),
                    )
                )
                sweep_root = zarr.open_group(store=sweep_store, mode="w")
                sweep_group = sweep_root.create_group("RNA")
                sweep_counts, _sweep_plan = create_candidate_counts(
                    sweep_group,
                    values,
                    policy=SCALED_POLICY,
                    profile="cloud",
                    resources=budget,
                )
                sweep_store.probe.reset()
                sweep_metrics: dict[str, Any] = {}
                sweep_started = time.perf_counter()
                sweep_counts_t = write_layout_counts_t(
                    sweep_counts,
                    sweep_group,
                    writer="asyncCandidate",
                    policy=SCALED_POLICY,
                    profile="cloud",
                    resources=budget,
                    readGroupChunks=1
                    if execution is None
                    else execution.readGroupChunks,
                    readGroupsInFlight=reads_in_flight,
                    destinationCommitsInFlight=commits_in_flight,
                    metrics=sweep_metrics,
                )
                np.testing.assert_array_equal(
                    np.asarray(sweep_counts_t[:]),
                    values.T,
                )
                sweep_observations[
                    f"reads-{reads_in_flight}-commits-{commits_in_flight}"
                ] = {
                    "uri": sweep_uri,
                    "seconds": time.perf_counter() - sweep_started,
                    "writer": sweep_metrics,
                    "storeOperations": sweep_store.probe.to_json(),
                    "kind": "observed",
                }
            observations["remote"]["outerConcurrencySweeps"] = sweep_observations
            failure_uri = config.phaseFailureStoreUri("phase2")
            failure_probe = StoreProbe(fail_on=chunk_writes[0])
            failure_store = wrap_recording_store(
                make_store(
                    failure_uri,
                    storage_options=storage_options(failure_uri),
                ),
                probe=failure_probe,
            )
            failure_root = zarr.open_group(store=failure_store, mode="w")
            failure_group = failure_root.create_group("RNA")
            failure_counts, _failure_plan = create_candidate_counts(
                failure_group,
                values,
                policy=SCALED_POLICY,
                profile="cloud",
                resources=budget,
            )
            failure_metrics: dict[str, Any] = {}
            try:
                write_layout_counts_t(
                    failure_counts,
                    failure_group,
                    writer="asyncCandidate",
                    policy=SCALED_POLICY,
                    profile="cloud",
                    resources=budget,
                    readGroupChunks=1
                    if execution is None
                    else execution.readGroupChunks,
                    readGroupsInFlight=1
                    if execution is None
                    else execution.readGroupsInFlight,
                    destinationCommitsInFlight=1
                    if execution is None
                    else execution.destinationCommitsInFlight,
                    metrics=failure_metrics,
                )
            except Exception as exc:
                nested = (
                    tuple(exc.exceptions) if isinstance(exc, BaseExceptionGroup) else ()
                )
                if "injected write failure" not in str(exc) and not any(
                    "injected write failure" in str(item) for item in nested
                ):
                    raise
            else:
                raise AssertionError("Phase 2 R2 failure injection did not propagate")
            failed_counts_t = failure_group["countsT"]
            if failed_counts_t.attrs.get("complete") is not False:
                raise AssertionError("Phase 2 R2 failed countsT was marked complete")
            if failure_metrics.get("heldLedgerBytes") != 0:
                raise AssertionError("Phase 2 R2 failure leaked admitted bytes")
            observations["remoteFailure"] = {
                "uri": failure_uri,
                "failureKey": chunk_writes[0],
                "complete": False,
                "writer": failure_metrics,
                "storeOperations": failure_probe.to_json(),
                "kind": "observed",
            }
            checks.extend(
                (
                    "r2-async-transpose",
                    "r2-writer-admission",
                    "r2-source-decode-accounting",
                    "r2-read-width-full-shard",
                    "r2-outer-concurrency-sweep",
                    "r2-controlled-failure",
                )
            )
        return PhaseWorkerResult(
            phase="phase2",
            hypothesis="H2",
            status="ok",
            checks=tuple(checks),
            deferredChecks=(),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseWorkerResult(
            phase="phase2",
            hypothesis="H2",
            status="error",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase2_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase2"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase2 worker failed")
        checks.append("worker-result-present")
        uri = config.phaseSyntheticStoreUri("phase2")
        if object_exists(f"{uri.rstrip('/')}/zarr.json"):
            from scarf.storage.stores import open_store

            from profiling.lattice_experiment import array_checksum, scaled_values
            from profiling.r2 import storage_options

            root = open_store(uri, mode="r", storage_options=storage_options(uri))
            values = scaled_values()
            loaded = np.asarray(root["RNA"]["countsT"][:])
            np.testing.assert_array_equal(loaded, values.T)
            if array_checksum(loaded) != worker.observations.get("local", {}).get(
                "checksum"
            ):
                raise AssertionError("reopened countsT checksum mismatch")
            checks.append("reopen-checksum")
            failure_uri = config.phaseFailureStoreUri("phase2")
            if not object_exists(f"{failure_uri.rstrip('/')}/zarr.json"):
                raise AssertionError("Phase 2 controlled-failure store is missing")
            failure_root = open_store(
                failure_uri,
                mode="r",
                storage_options=storage_options(failure_uri),
            )
            failure_counts_t = failure_root["RNA"]["countsT"]
            if failure_counts_t.attrs.get("complete") is not False:
                raise AssertionError("reopened failed countsT was marked complete")
            checks.append("reopen-controlled-failure-incomplete")
            for reads_in_flight, commits_in_flight in ((2, 1), (1, 2)):
                sweep_uri = config.phaseSweepStoreUri(
                    "phase2",
                    readGroupsInFlight=reads_in_flight,
                    destinationCommitsInFlight=commits_in_flight,
                )
                sweep_root = open_store(
                    sweep_uri,
                    mode="r",
                    storage_options=storage_options(sweep_uri),
                )
                np.testing.assert_array_equal(
                    np.asarray(sweep_root["RNA"]["countsT"][:]),
                    values.T,
                )
            checks.append("reopen-concurrency-sweeps")
        return PhaseReopenResult(
            phase="phase2",
            hypothesis="H2",
            status="ok",
            validated=True,
            checks=tuple(checks),
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseReopenResult(
            phase="phase2",
            hypothesis="H2",
            status="error",
            validated=False,
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase2(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    if worker.status != "ok" or reopen.status != "ok" or not reopen.validated:
        return PhaseFinalResult(
            phase="phase2",
            hypothesis="H2",
            status="error",
            decision="reject",
            branch="stop-writer",
            nextPhase=None,
            completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
            deferredChecks=worker.deferredChecks,
            worker=worker,
            reopen=reopen,
            error=worker.error or reopen.error,
        )
    return PhaseFinalResult(
        phase="phase2",
        hypothesis="H2",
        status="ok",
        decision="accept",
        branch="zarr-standard-store",
        nextPhase="phase3",
        completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
        deferredChecks=(),
        worker=worker,
        reopen=reopen,
    )


def phase3_variant_run_tag(
    config: ProfilingConfig,
    repetition: int,
    variant: Phase3VariantName,
    *,
    nRows: int = 100_000,
    namespace: ComparisonNamespace = "phase3",
) -> str:
    if namespace == "phase3":
        return f"{config.runTag}-phase3-r{int(repetition)}-{variant}"
    return f"{config.runTag}-scale-{int(nRows)}-r{int(repetition)}-{variant}"


def _phase3_variant_settings(
    variant: Phase3VariantName,
) -> tuple[str, str]:
    if variant == "currentWholeStrip":
        return "current", "wholeStrip"
    if variant == "currentBounded":
        return "current", "bounded"
    if variant == "candidateBounded":
        return "experimental", "bounded"
    raise ValueError(f"unsupported Phase 3 variant: {variant}")


def validate_phase3_prerequisites(config: ProfilingConfig) -> dict[str, Any]:
    """Validate durable Phase 0 evidence before scheduling 100k variants."""
    if 100_000 not in config.targetSizes:
        raise ValueError("Phase 3 requires 100000 in targetSizes")
    manifest = get_json(config.phaseInputManifestUri("phase0", 100_000))
    if manifest.get("status") != "ok" or int(manifest.get("nRows", 0)) != 100_000:
        raise RuntimeError("Phase 3 requires a validated 100k input manifest")
    if manifest.get("inferredStorageDtype") != "uint16":
        raise RuntimeError("Phase 3 requires the recorded 100k uint16 input policy")
    if not manifest.get("sha256"):
        raise RuntimeError("Phase 3 requires a SHA-256 for the 100k input")
    n_columns = int(manifest.get("nColumns", 0))
    policy = _policy_from_config(config)
    if n_columns < 1:
        raise RuntimeError("Phase 3 requires a positive input feature count")
    inventory = get_json(config.phaseClusterInventoryUri("phase0"))
    cluster = inventory.get("100000", {})
    candidates = [
        item
        for item in cluster.get("candidates", [])
        if item.get("status") == "candidate"
    ]
    if cluster.get("h7") != "accepted" or not candidates:
        raise RuntimeError("Phase 3 requires a validated 100k cluster source")
    selected = candidates[0]
    return {
        "inputSha256": manifest["sha256"],
        "nRows": 100_000,
        "nColumns": n_columns,
        "inferredStorageDtype": manifest["inferredStorageDtype"],
        "targetReadUnitBytes": policy.targetReadUnitBytes,
        "clusterSourceUri": selected.get("uri"),
        "clusterLabelColumn": selected.get("labelColumn"),
        "kind": "observed",
    }


def _phase3_workflow(
    config: ProfilingConfig,
    variant: Phase3VariantName,
    *,
    nRows: int = 100_000,
) -> Any:
    writer, consumer = _phase3_variant_settings(variant)
    cluster_source = config.clusterSourceFor(nRows)
    if cluster_source is None:
        inventory = get_json(config.phaseClusterInventoryUri("phase0"))
        candidates = inventory.get(str(int(nRows)), {}).get("candidates", [])
        selected = next(
            (item for item in candidates if item.get("status") == "candidate"),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"Comparison requires a validated {int(nRows)}-row cluster source"
            )
        cluster_uri = str(selected["uri"])
        label_column = str(selected.get("labelColumn", "RNA_leiden_cluster"))
    else:
        cluster_uri = cluster_source.storeUri
        label_column = cluster_source.labelColumn
    return config.workflow.model_copy(
        update={
            "countMatrixWriter": writer,
            "featureConsume": consumer,
            "clusterSourceUri": cluster_uri,
            "clusterLabelColumn": label_column,
        }
    )


def _digest_array_payload(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.dtype.kind in {"O", "U", "S"}:
        for item in array.reshape(-1):
            payload = (
                bytes(item)
                if isinstance(item, bytes | bytearray | np.bytes_)
                else str(item).encode()
            )
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _digest_zarr_group(group: Any) -> str:
    digest = hashlib.sha256()

    def _visit(current: Any, prefix: str) -> None:
        digest.update(prefix.encode())
        stable_attributes = {
            name: current.attrs[name]
            for name in (
                "group_order",
                "groups",
                "stat_columns",
                "n_group",
                "n_reference",
                "shape",
                "dims",
                "columns",
            )
            if name in current.attrs
        }
        digest.update(
            json.dumps(
                stable_attributes,
                sort_keys=True,
                default=str,
            ).encode()
        )
        for name in sorted(str(value) for value in current.array_keys()):
            array = current[name]
            digest.update(f"{prefix}/{name}".encode())
            digest.update(_digest_array_payload(np.asarray(array[:])).encode())
        for name in sorted(str(value) for value in current.group_keys()):
            _visit(current[name], f"{prefix}/{name}")

    _visit(group, "")
    return digest.hexdigest()


def _phase3_comparable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    array_digests = summary.get("arrayDigests", {})
    return {
        "orderedCellDigest": summary.get("orderedCellDigest"),
        "orderedFeatureDigest": summary.get("orderedFeatureDigest"),
        "exactArrayDigests": {
            name: digest
            for name, digest in array_digests.items()
            if not str(name).startswith("stats_")
        },
        "artifactDigests": {
            kind: sorted(str(item["digest"]) for item in records)
            for kind, records in summary.get("artifacts", {}).items()
            if kind == "marker_table"
        },
    }


def collect_phase3_outputs(
    storeUri: str,
    workflow: Any,
    resources: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from scarf import DataStore
    from scarf.storage.count_matrix import allow_experimental_layout_reads
    from scarf.storage.refs import artifact_path

    from profiling.r2 import storage_options

    arguments = {
        "nthreads": resources.workers,
        "zarr_mode": "r+",
        "zarrProfile": "cloud",
        "storage_options": storage_options(storeUri),
    }
    if workflow.countMatrixWriter == "experimental":
        with allow_experimental_layout_reads():
            store = DataStore(storeUri, **arguments)
    else:
        store = DataStore(storeUri, **arguments)
    assay = store._get_assay(workflow.assayName)
    assay._experimentalFeatureConsume = workflow.featureConsume
    context = (
        allow_experimental_layout_reads()
        if workflow.countMatrixWriter == "experimental"
        else None
    )

    def _collect() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        stats = assay.get_feature_stats(
            workflow.cellKey,
            columns=("normed_tot", "avg", "nz_mean", "sigmas", "normed_n"),
        )
        for name, values in assay.get_feature_stats(workflow.cellKey).items():
            stats.setdefault(name, values)
        arrays: dict[str, np.ndarray] = {
            "hvgs": np.asarray(assay.feats.fetch_all(workflow.resolvedHvgKey)),
            **{f"stats_{name}": np.asarray(values) for name, values in stats.items()},
        }
        artifacts: dict[str, list[dict[str, Any]]] = {}
        for kind in ("reduction", "marker_table"):
            records: list[dict[str, Any]] = []
            refs = store.list_artifacts(
                kind=kind,
                from_assay=workflow.assayName,
                scope="assay",
                complete_only=True,
            )
            for ref in refs:
                path = artifact_path(ref)
                records.append(
                    {
                        "artifactId": ref.artifact_id,
                        "digest": _digest_zarr_group(store.z[path]),
                    }
                )
            artifacts[kind] = records
        if not artifacts["reduction"]:
            raise AssertionError("Phase 3 produced no complete PCA reduction")
        if not artifacts["marker_table"]:
            raise AssertionError("Phase 3 produced no complete marker table")
        summary = {
            "orderedCellDigest": _ordered_id_digest(
                np.asarray(assay.cells.fetch_all("ids"))
            ),
            "orderedFeatureDigest": _ordered_id_digest(
                np.asarray(assay.feats.fetch_all("ids"))
            ),
            "arrayDigests": {
                name: _digest_array_payload(values) for name, values in arrays.items()
            },
            "artifacts": artifacts,
            "kind": "observed",
        }
        return summary, arrays

    if context is None:
        return _collect()
    with context:
        return _collect()


def validate_phase3_counts_t(
    storeUri: str,
    workflow: Any,
) -> dict[str, Any]:
    from scarf.storage.stores import open_store
    from scarf.storage.types import as_zarr_array, as_zarr_group

    from profiling.r2 import storage_options

    root = open_store(
        storeUri,
        mode="r",
        storage_options=storage_options(storeUri),
    )
    group = as_zarr_group(root[workflow.assayName], name=workflow.assayName)
    counts_name = (
        "countsCandidate" if workflow.countMatrixWriter == "experimental" else "counts"
    )
    counts = as_zarr_array(
        group[counts_name],
        name=f"{workflow.assayName}/{counts_name}",
    )
    counts_t = as_zarr_array(
        group["countsT"],
        name=f"{workflow.assayName}/countsT",
    )
    digest = hashlib.sha256()
    checked_tiles = 0
    for feat_start in range(0, int(counts_t.shape[0]), int(counts_t.chunks[0])):
        feat_end = min(
            feat_start + int(counts_t.chunks[0]),
            int(counts_t.shape[0]),
        )
        for cell_start in range(
            0,
            int(counts_t.shape[1]),
            int(counts_t.chunks[1]),
        ):
            cell_end = min(
                cell_start + int(counts_t.chunks[1]),
                int(counts_t.shape[1]),
            )
            actual = np.asarray(counts_t[feat_start:feat_end, cell_start:cell_end])
            expected = np.asarray(counts[cell_start:cell_end, feat_start:feat_end]).T
            np.testing.assert_array_equal(actual, expected)
            digest.update(np.ascontiguousarray(actual).tobytes())
            checked_tiles += 1
    return {
        "checkedTiles": checked_tiles,
        "checksum": digest.hexdigest(),
        "complete": counts_t.attrs.get("complete") is True,
        "kind": "observed",
    }


def run_phase3_variant_body(
    config: ProfilingConfig,
    repetition: int,
    variant: Phase3VariantName,
    workDir: Path,
    *,
    nRows: int = 100_000,
    namespace: ComparisonNamespace = "phase3",
) -> Phase3VariantResult:
    from scarf.storage.budget import ResourceBudget
    from scarf.storage.stores import open_store
    from scarf.storage.types import as_zarr_array, as_zarr_group

    from profiling.lattice_experiment import copy_candidate_counts
    from profiling.datasets import sha256_file
    from profiling.r2 import download_file, storage_options
    from profiling.stages import run_stage

    if nRows < 1:
        raise ValueError("nRows must be positive")
    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    child = config.model_copy(
        update={
            "runTag": phase3_variant_run_tag(
                config,
                repetition,
                variant,
                nRows=nRows,
                namespace=namespace,
            )
        }
    )
    store_uri = child.storeUri(nRows)
    if namespace == "phase3":
        result_uri = config.phase3ValidationUri(repetition, variant)
        reference_arrays_uri = config.phase3ReferenceArraysUri()
        reference_uri = config.phase3ReferenceUri()
    else:
        result_uri = config.scaleVariantResultUri(nRows, repetition, variant)
        reference_arrays_uri = config.scaleReferenceArraysUri(nRows)
        reference_uri = config.scaleReferenceUri(nRows)
    stages: dict[str, dict[str, Any]] = {}
    setup: dict[str, Any] = {
        "freshStore": True,
        "freshContainer": True,
        "cacheReuse": False,
        "modalRegion": config.modalRegion,
        "executionPolicy": (
            None
            if config.executionPolicy is None
            else config.executionPolicy.model_dump(mode="json")
        ),
        "resources": {
            stage: child.resourcesFor(stage).model_dump(mode="json")
            for stage in child.effectiveStages
        },
        "kind": "observed",
    }
    outputs: dict[str, Any] = {}
    error: str | None = None
    status: PhaseStatus = "ok"
    workDir.mkdir(parents=True, exist_ok=True)
    local_h5ad = workDir / f"{int(nRows)}.h5ad"

    def _run(stage: Any, *, localPath: Path | None = None) -> None:
        result = run_stage(
            stage,
            nRows=nRows,
            storeUri=store_uri,
            workflow=workflow,
            resources=child.resourcesFor(stage),
            localH5adPath=localPath,
            storageLayout=child.storageLayout,
            countMatrixLayout=child.countMatrixLayout,
            executionPolicy=child.executionPolicy,
            workDir=workDir / stage,
            invalidateCache=False,
            recordStoreOperations=True,
            clientProvenance=child.clientProvenance,
        )
        payload = result.to_json()
        if not put_json_if_absent(child.resultUri(nRows, stage), payload):
            raise FileExistsError(
                f"Phase 3 stage result already exists for {child.runTag}/{stage}"
            )
        stages[str(stage)] = payload
        if result.status != "ok":
            raise RuntimeError(result.error or f"{stage} failed")

    try:
        workflow = _phase3_workflow(config, variant, nRows=nRows)
        setup["workflow"] = workflow.model_dump(mode="json")
        setup["nRows"] = int(nRows)
        setup["namespace"] = namespace
        if object_exists(f"{store_uri.rstrip('/')}/zarr.json"):
            raise FileExistsError(f"Comparison store already exists: {store_uri}")
        download_file(config.datasetUri(nRows), local_h5ad)
        setup["inputSha256"] = sha256_file(local_h5ad)
        _run("createStore", localPath=local_h5ad)
        if variant == "candidateBounded":
            root = open_store(
                store_uri,
                mode="r+",
                storage_options=storage_options(store_uri),
            )
            group = as_zarr_group(
                root[workflow.assayName],
                name=workflow.assayName,
            )
            source = as_zarr_array(
                group["counts"],
                name=f"{workflow.assayName}/counts",
            )
            _candidate, candidate_details = copy_candidate_counts(
                group,
                source,
                policy=_policy_from_config(config),
                profile="cloud",
                resources=ResourceBudget(
                    child.resourcesFor("writeCountsT").scarfMemoryBudget,
                    child.resourcesFor("writeCountsT").workers,
                ),
            )
            setup["candidateCounts"] = candidate_details
        for stage in (
            "writeCountsT",
            "initializeStore",
            "reopenStore",
            "filterCells",
            "markHvgs",
            "runNormalization",
            "runPca",
            "importClusters",
            "findMarkers",
        ):
            _run(stage)
        outputs, reference_arrays = collect_phase3_outputs(
            store_uri,
            workflow,
            child.resourcesFor("reopenStore"),
        )
        if repetition == 0 and variant == "currentWholeStrip":
            reference_buffer = io.BytesIO()
            np.savez_compressed(reference_buffer, **reference_arrays)
            if not put_bytes_if_absent(
                reference_arrays_uri,
                reference_buffer.getvalue(),
            ):
                raise FileExistsError("Comparison reference arrays already exist")
            if not put_json_if_absent(reference_uri, outputs):
                raise FileExistsError("Comparison reference summary already exists")
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    result = Phase3VariantResult(
        repetition=repetition,
        variant=variant,
        runTag=child.runTag,
        storeUri=store_uri,
        status=status,
        stages=stages,
        setup=setup,
        outputs=outputs,
        error=error,
        provenance=provenance_from_config(child),
    )
    if not put_json_if_absent(result_uri, result.model_dump(mode="json")):
        raise FileExistsError(
            f"Comparison result already exists for {repetition}/{variant}"
        )
    return result


def _phase3_summaries(
    variants: list[Phase3VariantResult],
) -> dict[str, dict[str, Any]]:
    from profiling.lattice_experiment import median

    summaries: dict[str, dict[str, Any]] = {}
    for variant in (
        "currentWholeStrip",
        "currentBounded",
        "candidateBounded",
    ):
        selected = [
            item for item in variants if item.variant == variant and item.status == "ok"
        ]
        if not selected:
            continue
        writes = [float(item.stages["writeCountsT"]["seconds"]) for item in selected]
        markers = [float(item.stages["findMarkers"]["seconds"]) for item in selected]
        hvgs = [float(item.stages["markHvgs"]["seconds"]) for item in selected]
        pcas = [float(item.stages["runPca"]["seconds"]) for item in selected]
        primary_stage_names = ("writeCountsT", "findMarkers")

        def _stage_memory(stage: dict[str, Any]) -> int:
            return int(
                stage.get("operationIncrementalPeakBytes")
                or stage.get("peakCgroupBytes")
                or stage.get("peakRssBytes")
                or 0
            )

        memory = max(
            _stage_memory(stage)
            for item in selected
            for stage_name, stage in item.stages.items()
            if stage_name in primary_stage_names
        )
        hvg_memory = max(_stage_memory(item.stages["markHvgs"]) for item in selected)
        requested_read_bytes = sum(
            int(
                item.stages[stage_name]
                .get("details", {})
                .get("storeOperations", {})
                .get("readRequestedBytes", 0)
            )
            for item in selected
            for stage_name in ("writeCountsT", "findMarkers")
        )
        write_details = selected[0].stages["writeCountsT"].get("details", {})
        shape = tuple(int(value) for value in write_details.get("shape", ()))
        dtype = np.dtype(write_details.get("dtype", "uint16"))
        matrix_bytes = (
            int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
            if len(shape) == 2
            else 0
        )
        useful_bytes = matrix_bytes * 2 * len(selected)
        io_efficiency = (
            float(useful_bytes / requested_read_bytes)
            if requested_read_bytes > 0
            else 0.0
        )

        def _high_variance(values: list[float]) -> bool:
            center = median(values)
            if center <= 0:
                return max(values) != min(values)
            return (max(values) - min(values)) / center > 0.15

        summaries[variant] = {
            "reps": len(selected),
            "writeMedianSeconds": median(writes),
            "markerMedianSeconds": median(markers),
            "hvgMedianSeconds": median(hvgs),
            "pcaMedianSeconds": median(pcas),
            "writeSeconds": writes,
            "markerSeconds": markers,
            "hvgSeconds": hvgs,
            "pcaSeconds": pcas,
            "highVariance": any(
                _high_variance(values) for values in (writes, markers, hvgs)
            ),
            "peakMemoryBytes": memory,
            "hvgPeakMemoryBytes": hvg_memory,
            "requestedReadBytes": requested_read_bytes,
            "usefulLogicalBytes": useful_bytes,
            "usefulToRequestedBytes": io_efficiency,
            "writerAdmissionFailed": any(
                "MemoryError" in (item.error or "")
                for item in variants
                if item.variant == variant
            ),
            "kind": "observed",
        }
    return summaries


def build_scale_comparison_schedule(
    *,
    repetitions: int = 3,
    randomSeed: int = 4_466,
) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    variants: tuple[Phase3VariantName, ...] = (
        "currentWholeStrip",
        "currentBounded",
        "candidateBounded",
    )
    schedule: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        ordered = list(variants)
        random.Random(randomSeed + repetition).shuffle(ordered)
        if repetition == 0 and ordered[0] != "currentWholeStrip":
            ordered.remove("currentWholeStrip")
            ordered.insert(0, "currentWholeStrip")
        schedule.extend(
            {
                "repetition": repetition,
                "variant": variant,
                "order": len(schedule),
                "batch": "pilot" if repetition == 0 else "continuation",
            }
            for variant in ordered
        )
    return schedule


def validate_scale_comparison_prerequisites(
    config: ProfilingConfig,
    *,
    nRows: int,
    evidenceRunTag: str,
) -> dict[str, Any]:
    if nRows not in config.targetSizes:
        raise ValueError(f"Scale comparison requires {nRows} in targetSizes")
    if not evidenceRunTag.strip():
        raise ValueError("evidenceRunTag must be non-empty")
    evidence = config.model_copy(update={"runTag": evidenceRunTag})
    manifest = get_json(evidence.phaseInputManifestUri("phase0", nRows))
    if manifest.get("status") != "ok" or int(manifest.get("nRows", 0)) != nRows:
        raise RuntimeError(
            f"Scale comparison requires a validated {nRows}-row input manifest"
        )
    if manifest.get("inferredStorageDtype") != "uint16":
        raise RuntimeError("Scale comparison requires the recorded uint16 input")
    if not manifest.get("sha256"):
        raise RuntimeError("Scale comparison input manifest lacks a SHA-256")
    n_columns = int(manifest.get("nColumns", 0))
    policy = _policy_from_config(config)
    if n_columns < 1:
        raise RuntimeError("Scale comparison requires a positive input feature count")
    inventory = get_json(evidence.phaseClusterInventoryUri("phase0"))
    cluster = inventory.get(str(int(nRows)), {})
    candidates = [
        item
        for item in cluster.get("candidates", [])
        if item.get("status") == "candidate"
    ]
    if cluster.get("h7") != "accepted" or not candidates:
        raise RuntimeError(
            f"Scale comparison requires a validated {nRows}-row cluster source"
        )
    selected = candidates[0]
    source = ClusterSourceRef(
        nRows=nRows,
        storeUri=str(selected["uri"]),
        labelColumn=str(selected.get("labelColumn", "RNA_leiden_cluster")),
    )
    return {
        "inputSha256": manifest["sha256"],
        "nRows": nRows,
        "nColumns": n_columns,
        "inferredStorageDtype": manifest["inferredStorageDtype"],
        "targetReadUnitBytes": policy.targetReadUnitBytes,
        "clusterSource": source.model_dump(mode="json"),
        "evidenceRunTag": evidenceRunTag,
        "kind": "observed",
    }


def claim_scale_comparison(
    config: ProfilingConfig,
    *,
    nRows: int,
    evidenceRunTag: str,
    baselineRunTag: str,
    repetitions: int = 3,
    pilotRepetitions: int = 1,
) -> ScaleComparisonClaim:
    if not config.runTag.strip():
        raise ValueError("scale-check requires a non-empty runTag")
    if not baselineRunTag.strip():
        raise ValueError("baselineRunTag must be non-empty")
    provenance = provenance_from_config(config)
    existing_uri = config.scaleClaimUri(nRows)
    if object_exists(existing_uri):
        existing = ScaleComparisonClaim.model_validate(get_json(existing_uri))
        expected = {
            "runTag": config.runTag,
            "nRows": nRows,
            "repetitions": repetitions,
            "pilotRepetitions": pilotRepetitions,
            "evidenceRunTag": evidenceRunTag,
            "baselineRunTag": baselineRunTag,
            "decisionRule": "scale-focused",
            "sourceTreeSha256": provenance.get("sourceTreeSha256"),
            "lockfileSha256": provenance.get("lockfileSha256"),
            "configSha256": provenance.get("configSha256"),
        }
        actual = existing.model_dump(mode="json")
        actual.pop("claimedAt")
        if actual != expected:
            raise RuntimeError("Existing scale comparison claim does not match")
        return existing
    conflicts = [
        config.scaleScheduleUri(nRows),
        config.scaleReferenceUri(nRows),
        config.scaleReferenceArraysUri(nRows),
        config.scaleBatchValidationUri(nRows, "pilot"),
        config.scaleBatchValidationUri(nRows, "continuation"),
        config.scaleFinalResultUri(nRows),
        config.scaleCallReceiptUri(nRows, "validation-pilot"),
        config.scaleCallReceiptUri(nRows, "validation-continuation"),
        *(
            config.scaleVariantResultUri(nRows, repetition, variant)
            for repetition in range(repetitions)
            for variant in (
                "currentWholeStrip",
                "currentBounded",
                "candidateBounded",
            )
        ),
        *(
            config.scaleCallReceiptUri(
                nRows,
                f"variant-r{repetition}-{variant}",
            )
            for repetition in range(repetitions)
            for variant in (
                "currentWholeStrip",
                "currentBounded",
                "candidateBounded",
            )
        ),
    ]
    found = [uri for uri in conflicts if object_exists(uri)]
    if found:
        raise RuntimeError(
            "Refusing conflicting scale comparison destinations: " + ", ".join(found)
        )
    claim = ScaleComparisonClaim(
        runTag=config.runTag,
        nRows=nRows,
        repetitions=repetitions,
        pilotRepetitions=pilotRepetitions,
        evidenceRunTag=evidenceRunTag,
        baselineRunTag=baselineRunTag,
        decisionRule="scale-focused",
        claimedAt=utc_now(),
        sourceTreeSha256=provenance.get("sourceTreeSha256"),
        lockfileSha256=provenance.get("lockfileSha256"),
        configSha256=provenance.get("configSha256"),
    )
    _write_create_only(existing_uri, claim.model_dump(mode="json"))
    return claim


def _comparison_reference_uris(
    config: ProfilingConfig,
    *,
    nRows: int,
    namespace: ComparisonNamespace,
) -> tuple[str, str]:
    if namespace == "phase3":
        return config.phase3ReferenceUri(), config.phase3ReferenceArraysUri()
    return config.scaleReferenceUri(nRows), config.scaleReferenceArraysUri(nRows)


def _open_comparison_reduction(
    storeUri: str,
    workflow: Any,
    resources: Any,
) -> Any:
    from scarf import DataStore
    from scarf.storage.count_matrix import allow_experimental_layout_reads
    from scarf.storage.refs import artifact_path

    from profiling.r2 import storage_options

    arguments = {
        "nthreads": resources.workers,
        "zarr_mode": "r",
        "zarrProfile": "cloud",
        "storage_options": storage_options(storeUri),
    }
    if workflow.countMatrixWriter == "experimental":
        with allow_experimental_layout_reads():
            store = DataStore(storeUri, **arguments)
    else:
        store = DataStore(storeUri, **arguments)
    refs = store.list_artifacts(
        kind="reduction",
        from_assay=workflow.assayName,
        scope="assay",
        complete_only=True,
    )
    if len(refs) != 1:
        raise AssertionError(
            f"Expected one complete PCA reduction in {storeUri}, found {len(refs)}"
        )
    return store.z[artifact_path(refs[0])]


def _pca_tolerances(dtype: np.dtype[Any]) -> tuple[float, float]:
    if dtype == np.dtype("float32"):
        return PHASE3_PCA_FLOAT32_RTOL, PHASE3_PCA_FLOAT32_ATOL
    if dtype == np.dtype("float64"):
        return PHASE3_PCA_FLOAT64_RTOL, PHASE3_PCA_FLOAT64_ATOL
    raise TypeError(f"Unsupported PCA array dtype: {dtype}")


def _validate_pca_reduction(
    config: ProfilingConfig,
    *,
    reference: Phase3VariantResult,
    actual: Phase3VariantResult,
    nRows: int,
) -> dict[str, Any]:
    if actual.storeUri == reference.storeUri:
        return {
            "reference": True,
            "arrays": {},
            "kind": "observed",
        }
    resources = config.resourcesFor("reopenStore")
    reference_workflow = _phase3_workflow(
        config,
        reference.variant,
        nRows=nRows,
    )
    actual_workflow = _phase3_workflow(
        config,
        actual.variant,
        nRows=nRows,
    )
    reference_group = _open_comparison_reduction(
        reference.storeUri,
        reference_workflow,
        resources,
    )
    actual_group = _open_comparison_reduction(
        actual.storeUri,
        actual_workflow,
        resources,
    )
    reference_keys = sorted(str(name) for name in reference_group.array_keys())
    actual_keys = sorted(str(name) for name in actual_group.array_keys())
    if actual_keys != reference_keys:
        raise AssertionError("PCA reduction array names differ from the reference")
    array_results: dict[str, Any] = {}
    target_block_bytes = 64 * 1024**2
    for name in reference_keys:
        expected_array = reference_group[name]
        actual_array = actual_group[name]
        expected_dtype = np.dtype(expected_array.dtype)
        actual_dtype = np.dtype(actual_array.dtype)
        if tuple(actual_array.shape) != tuple(expected_array.shape):
            raise AssertionError(f"PCA {name} shape differs from the reference")
        if actual_dtype != expected_dtype:
            raise AssertionError(f"PCA {name} dtype differs from the reference")
        rtol, atol = _pca_tolerances(expected_dtype)
        shape = tuple(int(value) for value in expected_array.shape)
        row_values = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        rows_per_block = max(
            1,
            target_block_bytes // max(1, row_values * expected_dtype.itemsize),
        )
        unequal_values = 0
        max_abs_difference = 0.0
        for start in range(0, shape[0], rows_per_block):
            stop = min(shape[0], start + rows_per_block)
            expected = np.asarray(expected_array[start:stop])
            observed = np.asarray(actual_array[start:stop])
            try:
                np.testing.assert_allclose(
                    observed,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"PCA {name} differs beyond tolerance in rows {start}:{stop}; "
                    f"rtol={rtol}, atol={atol}"
                ) from exc
            unequal_values += int(
                np.count_nonzero(
                    (observed != expected) & ~(np.isnan(observed) & np.isnan(expected))
                )
            )
            differences = np.abs(
                observed.astype(np.float64) - expected.astype(np.float64)
            )
            if differences.size and not np.all(np.isnan(differences)):
                max_abs_difference = max(
                    max_abs_difference,
                    float(np.nanmax(differences)),
                )
        array_results[name] = {
            "shape": list(shape),
            "dtype": expected_dtype.name,
            "rtol": rtol,
            "atol": atol,
            "unequalValues": unequal_values,
            "maxAbsDifference": max_abs_difference,
            "kind": "observed",
        }
    return {
        "reference": False,
        "arrays": array_results,
        "kind": "observed",
    }


def _validate_comparison_outputs(
    config: ProfilingConfig,
    variants: list[Phase3VariantResult],
    *,
    nRows: int,
    namespace: ComparisonNamespace,
) -> list[dict[str, Any]]:
    reference_uri, reference_arrays_uri = _comparison_reference_uris(
        config,
        nRows=nRows,
        namespace=namespace,
    )
    reference_summary = get_json(reference_uri)
    with np.load(io.BytesIO(get_bytes(reference_arrays_uri))) as data:
        reference_arrays = {name: np.asarray(data[name]) for name in data.files}
    if namespace == "phase3":
        reference_variant_uri = config.phase3ValidationUri(
            0,
            "currentWholeStrip",
        )
    else:
        reference_variant_uri = config.scaleVariantResultUri(
            nRows,
            0,
            "currentWholeStrip",
        )
    reference_variant = Phase3VariantResult.model_validate(
        get_json(reference_variant_uri)
    )
    validations: list[dict[str, Any]] = []
    for variant in variants:
        workflow = _phase3_workflow(config, variant.variant, nRows=nRows)
        actual_summary, actual_arrays = collect_phase3_outputs(
            variant.storeUri,
            workflow,
            config.resourcesFor("reopenStore"),
        )
        transpose = validate_phase3_counts_t(variant.storeUri, workflow)
        if not transpose["complete"]:
            raise AssertionError(f"countsT is incomplete for {variant.runTag}")
        if actual_summary != variant.outputs:
            raise AssertionError(
                f"reopened output summary changed for {variant.runTag}"
            )
        for name, expected in reference_arrays.items():
            actual = actual_arrays[name]
            if name.startswith("stats_"):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=PHASE3_HVG_RTOL,
                    atol=PHASE3_HVG_ATOL,
                    equal_nan=True,
                )
            else:
                try:
                    np.testing.assert_array_equal(actual, expected)
                except AssertionError as exc:
                    if name == "hvgs":
                        corrected_names = [
                            key
                            for key in reference_arrays
                            if key.startswith("stats_c_var")
                        ]
                        if corrected_names:
                            corrected = reference_arrays[corrected_names[0]]
                            selected = np.asarray(expected).astype(bool)
                            cutoff = (
                                float(np.min(corrected[selected]))
                                if np.any(selected)
                                else float("nan")
                            )
                            raise AssertionError(
                                "HVG IDs differ; cutoff="
                                f"{cutoff}, tieBand={PHASE3_HVG_CUTOFF_TIE_BAND}"
                            ) from exc
                    raise
        if _phase3_comparable_summary(actual_summary) != _phase3_comparable_summary(
            reference_summary
        ):
            raise AssertionError(
                f"comparison outputs differ from reference for {variant.runTag}"
            )
        pca_reduction = _validate_pca_reduction(
            config,
            reference=reference_variant,
            actual=variant,
            nRows=nRows,
        )
        validations.append(
            {
                "runTag": variant.runTag,
                "variant": variant.variant,
                "repetition": variant.repetition,
                "validated": True,
                "transpose": transpose,
                "pcaReduction": pca_reduction,
            }
        )
    return validations


def run_scale_batch_validation_body(
    config: ProfilingConfig,
    *,
    nRows: int,
    batch: Literal["pilot", "continuation"],
    repetitionStart: int,
    repetitionEnd: int,
    expectedInputSha256: str,
) -> ScaleBatchValidationResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        if repetitionStart < 0 or repetitionEnd <= repetitionStart:
            raise ValueError("invalid scale comparison repetition range")
        variants = [
            Phase3VariantResult.model_validate(
                get_json(config.scaleVariantResultUri(nRows, repetition, variant))
            )
            for repetition in range(repetitionStart, repetitionEnd)
            for variant in (
                "currentWholeStrip",
                "currentBounded",
                "candidateBounded",
            )
        ]
        expected_count = (repetitionEnd - repetitionStart) * 3
        if len(variants) != expected_count or any(
            item.status != "ok" for item in variants
        ):
            errors = [item.error for item in variants if item.status != "ok"]
            raise AssertionError(
                "Scale comparison batch has unsuccessful variants: "
                + "; ".join(str(item) for item in errors)
            )
        run_tags = {item.runTag for item in variants}
        store_uris = {item.storeUri for item in variants}
        modal_inputs = {
            str((item.provenance or {}).get("modalInputId", "")) for item in variants
        }
        if len(run_tags) != expected_count or len(store_uris) != expected_count:
            raise AssertionError("Scale comparison variants reused destinations")
        if "" in modal_inputs or len(modal_inputs) != expected_count:
            raise AssertionError("Scale comparison variants did not use fresh inputs")
        expected_stages = {
            "createStore",
            "writeCountsT",
            "initializeStore",
            "reopenStore",
            "filterCells",
            "markHvgs",
            "runNormalization",
            "runPca",
            "importClusters",
            "findMarkers",
        }
        if any(set(item.stages) != expected_stages for item in variants):
            raise AssertionError("Scale comparison variant stages are incomplete")
        input_hashes = {str(item.setup.get("inputSha256")) for item in variants}
        if input_hashes != {expectedInputSha256}:
            raise AssertionError("Scale comparison variants used the wrong input")
        resource_payloads = {
            json.dumps(item.setup.get("resources"), sort_keys=True) for item in variants
        }
        if len(resource_payloads) != 1:
            raise AssertionError("Scale comparison variants used different resources")
        checks.extend(
            (
                "successful-variants",
                "fresh-variant-containers",
                "immutable-input",
                "identical-resources",
                "real-hvg-pca-marker-stages",
            )
        )
        validations = _validate_comparison_outputs(
            config,
            variants,
            nRows=nRows,
            namespace="scale",
        )
        checks.extend(
            (
                "independent-store-reopen",
                "exact-transpose",
                "exact-hvg-stats",
                "numerically-equivalent-pca-reduction",
                "exact-marker-outputs",
            )
        )
        return ScaleBatchValidationResult(
            nRows=nRows,
            batch=batch,
            repetitionStart=repetitionStart,
            repetitionEnd=repetitionEnd,
            status="ok",
            validated=True,
            checks=tuple(checks),
            validations=tuple(validations),
            provenance=provenance,
        )
    except Exception as exc:
        return ScaleBatchValidationResult(
            nRows=nRows,
            batch=batch,
            repetitionStart=repetitionStart,
            repetitionEnd=repetitionEnd,
            status="error",
            validated=False,
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_scale_comparison(
    config: ProfilingConfig,
    *,
    nRows: int,
    repetitions: int,
    baselineRunTag: str,
) -> ScaleComparisonFinalResult:
    provenance = provenance_from_config(config)
    pilot_uri = config.scaleBatchValidationUri(nRows, "pilot")
    continuation_uri = config.scaleBatchValidationUri(nRows, "continuation")
    try:
        pilot = ScaleBatchValidationResult.model_validate(get_json(pilot_uri))
        continuation = ScaleBatchValidationResult.model_validate(
            get_json(continuation_uri)
        )
        if not pilot.validated or pilot.status != "ok":
            raise RuntimeError(pilot.error or "Scale comparison pilot failed")
        if not continuation.validated or continuation.status != "ok":
            raise RuntimeError(
                continuation.error or "Scale comparison continuation failed"
            )
        variants = [
            Phase3VariantResult.model_validate(
                get_json(config.scaleVariantResultUri(nRows, repetition, variant))
            )
            for repetition in range(repetitions)
            for variant in (
                "currentWholeStrip",
                "currentBounded",
                "candidateBounded",
            )
        ]
        summaries = _phase3_summaries(variants)
        baseline_config = config.model_copy(update={"runTag": baselineRunTag})
        baseline_worker = PhaseWorkerResult.model_validate(
            get_json(baseline_config.phaseWorkerResultUri("phase3"))
        )
        baseline_summaries = baseline_worker.observations.get("summaries", {})
        if set(summaries) != set(baseline_summaries):
            raise AssertionError("Scale and 100k summaries have different variants")
        metrics = (
            "writeMedianSeconds",
            "markerMedianSeconds",
            "hvgMedianSeconds",
            "pcaMedianSeconds",
            "peakMemoryBytes",
            "hvgPeakMemoryBytes",
            "usefulToRequestedBytes",
        )
        scaling_context: dict[str, dict[str, Any]] = {
            "_comparison": {
                "kind": "descriptive",
                "warning": (
                    "100k and scale runs use different resource envelopes; "
                    "ratios are context, not a same-machine scaling curve"
                ),
            }
        }
        for variant, summary in summaries.items():
            baseline = baseline_summaries[variant]
            scaling_context[variant] = {
                metric: {
                    "baseline100k": baseline.get(metric),
                    "scale": summary.get(metric),
                    "ratio": (
                        float(summary[metric]) / float(baseline[metric])
                        if float(baseline.get(metric, 0) or 0) != 0
                        else None
                    ),
                }
                for metric in metrics
            }
        completed_checks = tuple(dict.fromkeys([*pilot.checks, *continuation.checks]))
        return ScaleComparisonFinalResult(
            nRows=nRows,
            status="ok",
            conclusion="measurement-complete",
            decisionRule="scale-focused",
            repetitions=repetitions,
            completedChecks=completed_checks,
            summaries=summaries,
            baseline100kSummaries=baseline_summaries,
            scalingContext=scaling_context,
            pilotValidationUri=pilot_uri,
            continuationValidationUri=continuation_uri,
            provenance=provenance,
        )
    except Exception as exc:
        return ScaleComparisonFinalResult(
            nRows=nRows,
            status="error",
            conclusion="measurement-failed",
            decisionRule="scale-focused",
            repetitions=repetitions,
            completedChecks=(),
            pilotValidationUri=pilot_uri,
            continuationValidationUri=continuation_uri,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase3_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.lattice_experiment import (
        run_phase3_local_checks,
        select_phase3_branch,
    )

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H3", "kind": "observed"}
    try:
        local = run_phase3_local_checks(reps=1)
        checks.extend(local["checks"])
        observations["local"] = local
        if includeRemote:
            variants = [
                Phase3VariantResult.model_validate(
                    get_json(config.phase3ValidationUri(repetition, variant))
                )
                for repetition in range(3)
                for variant in (
                    "currentWholeStrip",
                    "currentBounded",
                    "candidateBounded",
                )
            ]
            successful = [item for item in variants if item.status == "ok"]
            if len(successful) != 9:
                errors = [
                    item.error or f"{item.variant} repetition {item.repetition} failed"
                    for item in variants
                    if item.status != "ok"
                ]
                raise AssertionError(
                    "Phase 3 requires nine successful variants: " + "; ".join(errors)
                )
            expected_keys = {
                (repetition, variant)
                for repetition in range(3)
                for variant in (
                    "currentWholeStrip",
                    "currentBounded",
                    "candidateBounded",
                )
            }
            actual_keys = {(item.repetition, item.variant) for item in variants}
            if actual_keys != expected_keys:
                raise AssertionError("Phase 3 variant repetitions are incomplete")
            run_tags = {item.runTag for item in successful}
            store_uris = {item.storeUri for item in successful}
            if len(run_tags) != 9 or len(store_uris) != 9:
                raise AssertionError("Phase 3 variants did not use fresh destinations")
            modal_input_ids = {
                str((item.provenance or {}).get("modalInputId", ""))
                for item in successful
            }
            if "" in modal_input_ids or len(modal_input_ids) != 9:
                raise AssertionError(
                    "Phase 3 variants did not record nine fresh Modal inputs"
                )
            expected_stages = {
                "createStore",
                "writeCountsT",
                "initializeStore",
                "reopenStore",
                "filterCells",
                "markHvgs",
                "runNormalization",
                "runPca",
                "importClusters",
                "findMarkers",
            }
            if any(set(item.stages) != expected_stages for item in successful):
                raise AssertionError("Phase 3 variants did not run every real stage")
            input_hashes = {str(item.setup.get("inputSha256")) for item in successful}
            expected_input_hash = get_json(
                config.phaseInputManifestUri("phase0", 100_000)
            ).get("sha256")
            if input_hashes != {str(expected_input_hash)}:
                raise AssertionError(
                    "Phase 3 variants did not use the immutable 100k input"
                )
            resource_payloads = {
                json.dumps(item.setup.get("resources"), sort_keys=True)
                for item in successful
            }
            if len(resource_payloads) != 1:
                raise AssertionError("Phase 3 variants did not use identical resources")
            summaries = _phase3_summaries(variants)
            branch, reason = select_phase3_branch(summaries)
            observations["variants"] = [
                item.model_dump(mode="json") for item in variants
            ]
            observations["summaries"] = summaries
            observations["recordedBranch"] = branch
            observations["branchReason"] = reason
            schedule = get_json(config.phase3ScheduleUri())
            scheduled_keys = {
                (int(item["repetition"]), str(item["variant"]))
                for item in schedule.get("schedule", [])
            }
            if scheduled_keys != expected_keys:
                raise AssertionError("Phase 3 durable schedule is incomplete")
            observations["schedule"] = schedule
            checks.extend(
                (
                    "real-100k-input",
                    "three-repetitions",
                    "fresh-variant-containers",
                    "real-hvg-pca-marker-stages",
                    "branch-recorded",
                )
            )
        else:
            observations["recordedBranch"] = local["branch"]
            observations["branchReason"] = local["reason"]
        return PhaseWorkerResult(
            phase="phase3",
            hypothesis="H3",
            status="ok",
            checks=tuple(checks),
            deferredChecks=(),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseWorkerResult(
            phase="phase3",
            hypothesis="H3",
            status="error",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase3_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"kind": "observed"}
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase3"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase3 worker failed")
        checks.append("worker-result-present")
        branch = worker.observations.get("recordedBranch")
        if branch not in {"A", "B", "C", "D", "E"}:
            raise AssertionError("phase3 did not record a branch")
        checks.append("branch-recorded")
        if "variants" in worker.observations:
            reference_summary = get_json(config.phase3ReferenceUri())
            with np.load(
                io.BytesIO(get_bytes(config.phase3ReferenceArraysUri()))
            ) as data:
                reference_arrays = {name: np.asarray(data[name]) for name in data.files}
            reference_variant = next(
                Phase3VariantResult.model_validate(payload)
                for payload in worker.observations["variants"]
                if int(payload["repetition"]) == 0
                and str(payload["variant"]) == "currentWholeStrip"
            )
            validations: list[dict[str, Any]] = []
            for payload in worker.observations["variants"]:
                variant = Phase3VariantResult.model_validate(payload)
                if variant.status != "ok":
                    raise RuntimeError(
                        variant.error or f"Phase 3 variant failed: {variant.runTag}"
                    )
                workflow = _phase3_workflow(config, variant.variant)
                actual_summary, actual_arrays = collect_phase3_outputs(
                    variant.storeUri,
                    workflow,
                    config.resourcesFor("reopenStore"),
                )
                transpose = validate_phase3_counts_t(
                    variant.storeUri,
                    workflow,
                )
                if not transpose["complete"]:
                    raise AssertionError(f"countsT is incomplete for {variant.runTag}")
                if actual_summary != variant.outputs:
                    raise AssertionError(
                        f"reopened output summary changed for {variant.runTag}"
                    )
                for name, expected in reference_arrays.items():
                    actual = actual_arrays[name]
                    if name.startswith("stats_"):
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=PHASE3_HVG_RTOL,
                            atol=PHASE3_HVG_ATOL,
                            equal_nan=True,
                        )
                    else:
                        try:
                            np.testing.assert_array_equal(actual, expected)
                        except AssertionError as exc:
                            if name == "hvgs":
                                corrected_names = [
                                    key
                                    for key in reference_arrays
                                    if key.startswith("stats_c_var")
                                ]
                                if corrected_names:
                                    corrected = reference_arrays[corrected_names[0]]
                                    selected = np.asarray(expected).astype(bool)
                                    cutoff = (
                                        float(np.min(corrected[selected]))
                                        if np.any(selected)
                                        else float("nan")
                                    )
                                    raise AssertionError(
                                        "HVG IDs differ; cutoff="
                                        f"{cutoff}, tieBand="
                                        f"{PHASE3_HVG_CUTOFF_TIE_BAND}"
                                    ) from exc
                            raise
                if _phase3_comparable_summary(
                    actual_summary
                ) != _phase3_comparable_summary(reference_summary):
                    raise AssertionError(
                        f"Phase 3 outputs differ from reference for {variant.runTag}"
                    )
                pca_reduction = _validate_pca_reduction(
                    config,
                    reference=reference_variant,
                    actual=variant,
                    nRows=100_000,
                )
                validations.append(
                    {
                        "runTag": variant.runTag,
                        "variant": variant.variant,
                        "repetition": variant.repetition,
                        "validated": True,
                        "transpose": transpose,
                        "pcaReduction": pca_reduction,
                    }
                )
            observations["validations"] = validations
            checks.extend(
                (
                    "independent-store-reopen",
                    "exact-hvg-stats",
                    "numerically-equivalent-pca-reduction",
                    "exact-marker-outputs",
                )
            )
        return PhaseReopenResult(
            phase="phase3",
            hypothesis="H3",
            status="ok",
            validated=True,
            checks=tuple(checks),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return PhaseReopenResult(
            phase="phase3",
            hypothesis="H3",
            status="error",
            validated=False,
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase3(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    branch = (
        str(
            worker.observations.get(
                "recordedBranch",
                worker.observations.get("local", {}).get("branch", "E"),
            )
        )
        if worker.status == "ok"
        else "E"
    )
    if worker.status != "ok" or reopen.status != "ok" or not reopen.validated:
        return PhaseFinalResult(
            phase="phase3",
            hypothesis="H3",
            status="error",
            decision="reject",
            branch="E",
            nextPhase=None,
            completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
            deferredChecks=worker.deferredChecks,
            worker=worker,
            reopen=reopen,
            error=worker.error or reopen.error,
        )
    if branch in {"A", "B", "C"}:
        decision: PhaseDecision = "accept"
        next_phase: PhaseName | None = "phase4"
    elif branch == "D":
        decision = "reject"
        next_phase = None
    else:
        decision = "blocked"
        next_phase = None
    return PhaseFinalResult(
        phase="phase3",
        hypothesis="H3",
        status="ok" if decision != "reject" else "error",
        decision=decision,
        branch=branch,
        nextPhase=next_phase,
        completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
        deferredChecks=(),
        worker=worker,
        reopen=reopen,
    )


def _failed_worker(
    phase: PhaseName,
    hypothesis: HypothesisId,
    *,
    checks: tuple[str, ...],
    observations: dict[str, Any],
    error: str,
    provenance: dict[str, Any],
) -> PhaseWorkerResult:
    return PhaseWorkerResult(
        phase=phase,
        hypothesis=hypothesis,
        status="error",
        checks=checks,
        deferredChecks=(),
        observations=observations,
        error=error,
        provenance=provenance,
    )


def _ok_reopen(
    phase: PhaseName,
    hypothesis: HypothesisId,
    *,
    checks: tuple[str, ...],
    provenance: dict[str, Any],
) -> PhaseReopenResult:
    return PhaseReopenResult(
        phase=phase,
        hypothesis=hypothesis,
        status="ok",
        validated=True,
        checks=checks,
        provenance=provenance,
    )


def _failed_reopen(
    phase: PhaseName,
    hypothesis: HypothesisId,
    *,
    checks: tuple[str, ...],
    error: str,
    provenance: dict[str, Any],
) -> PhaseReopenResult:
    return PhaseReopenResult(
        phase=phase,
        hypothesis=hypothesis,
        status="error",
        validated=False,
        checks=checks,
        error=error,
        provenance=provenance,
    )


def _finalize_from_local(
    phase: PhaseName,
    hypothesis: HypothesisId,
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
    *,
    branch: str,
    next_phase: PhaseName | None,
    decision: PhaseDecision = "accept",
) -> PhaseFinalResult:
    if worker.status != "ok" or reopen.status != "ok" or not reopen.validated:
        return PhaseFinalResult(
            phase=phase,
            hypothesis=hypothesis,
            status="error",
            decision="reject",
            branch=branch,
            nextPhase=None,
            completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
            deferredChecks=worker.deferredChecks,
            worker=worker,
            reopen=reopen,
            error=worker.error or reopen.error,
        )
    return PhaseFinalResult(
        phase=phase,
        hypothesis=hypothesis,
        status="ok" if decision != "reject" else "error",
        decision=decision,
        branch=branch,
        nextPhase=next_phase,
        completedChecks=tuple(dict.fromkeys([*worker.checks, *reopen.checks])),
        deferredChecks=(),
        worker=worker,
        reopen=reopen,
    )


def run_phase4_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.lattice_experiment import run_phase4_local_checks

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H8", "kind": "observed"}
    workDir.mkdir(parents=True, exist_ok=True)
    try:
        local = run_phase4_local_checks()
        checks.extend(local["checks"])
        observations["local"] = local
        observations["productBranch"] = local["productBranch"]
        if includeRemote:
            observations["remote"] = {
                "status": "deferred-to-recorded-branch",
                "message": (
                    "Product producers stay on the current layout until Phase 3 "
                    "records Branch A from the 100k comparison"
                ),
                "kind": "derived",
            }
        return PhaseWorkerResult(
            phase="phase4",
            hypothesis="H8",
            status="ok",
            checks=tuple(checks),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return _failed_worker(
            "phase4",
            "H8",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase4_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase4"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase4 worker failed")
        checks.append("worker-result-present")
        if worker.observations.get("productBranch") != "current":
            raise AssertionError("phase4 must leave the product branch at current")
        checks.append("product-branch-current")
        return _ok_reopen("phase4", "H8", checks=tuple(checks), provenance=provenance)
    except Exception as exc:
        return _failed_reopen(
            "phase4",
            "H8",
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase4(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    return _finalize_from_local(
        "phase4",
        "H8",
        worker,
        reopen,
        branch=str(worker.observations.get("productBranch", "current")),
        next_phase="phase5",
    )


def run_phase5_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.lattice_experiment import run_phase5_local_checks

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H5", "kind": "observed"}
    workDir.mkdir(parents=True, exist_ok=True)
    try:
        local = run_phase5_local_checks()
        checks.extend(local["checks"])
        observations["local"] = local
        if includeRemote:
            observations["remote"] = local
            checks.append("scaled-consumer-oracle")
        return PhaseWorkerResult(
            phase="phase5",
            hypothesis="H5",
            status="ok",
            checks=tuple(checks),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return _failed_worker(
            "phase5",
            "H5",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase5_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase5"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase5 worker failed")
        checks.append("worker-result-present")
        local = worker.observations.get("local", {})
        if "unsorted-cell-order" not in local.get("checks", ()):
            raise AssertionError("phase5 did not preserve requested cell order")
        checks.append("cell-order-recorded")
        return _ok_reopen("phase5", "H5", checks=tuple(checks), provenance=provenance)
    except Exception as exc:
        return _failed_reopen(
            "phase5",
            "H5",
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase5(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    return _finalize_from_local(
        "phase5",
        "H5",
        worker,
        reopen,
        branch="bounded-consumers",
        next_phase="phase6",
    )


def run_phase6_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.lattice_experiment import run_phase6_local_checks

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H7", "kind": "observed"}
    workDir.mkdir(parents=True, exist_ok=True)
    try:
        local = run_phase6_local_checks()
        checks.extend(local["checks"])
        observations["local"] = local
        if includeRemote:
            observations["clusterSourcesPresent"] = CLUSTER_SOURCES_PATH.is_file()
            checks.append("cluster-source-inventory-path")
        return PhaseWorkerResult(
            phase="phase6",
            hypothesis="H7",
            status="ok",
            checks=tuple(checks),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return _failed_worker(
            "phase6",
            "H7",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase6_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase6"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase6 worker failed")
        checks.append("worker-result-present")
        return _ok_reopen("phase6", "H7", checks=tuple(checks), provenance=provenance)
    except Exception as rec_exc:
        return _failed_reopen(
            "phase6",
            "H7",
            checks=tuple(checks),
            error=f"{type(rec_exc).__name__}: {rec_exc}",
            provenance=provenance,
        )


def finalize_phase6(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    return _finalize_from_local(
        "phase6",
        "H7",
        worker,
        reopen,
        branch="imported-cluster-contract",
        next_phase="phase7",
    )


def run_phase7_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    from profiling.config import SELECTED_STAGE_ORDER, validate_requested_stages

    provenance = provenance_from_config(config)
    checks: list[str] = []
    observations: dict[str, Any] = {"hypothesis": "H4", "kind": "derived"}
    workDir.mkdir(parents=True, exist_ok=True)
    try:
        validate_requested_stages(SELECTED_STAGE_ORDER)
        checks.append("selected-stage-graph")
        observations["selectedStages"] = list(SELECTED_STAGE_ORDER)
        observations["oneMillionRequiresApproval"] = True
        if includeRemote:
            observations["remote"] = {
                "status": "blocked",
                "message": (
                    "10k and 100k selected-stage funnels use run-all with a fresh "
                    "runTag. 1M requires explicit approval."
                ),
                "kind": "derived",
            }
            checks.append("wave-one-requires-fresh-runtag")
        return PhaseWorkerResult(
            phase="phase7",
            hypothesis="H4",
            status="ok",
            checks=tuple(checks),
            observations=observations,
            provenance=provenance,
        )
    except Exception as exc:
        return _failed_worker(
            "phase7",
            "H4",
            checks=tuple(checks),
            observations=observations,
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def run_phase7_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase7"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase7 worker failed")
        checks.append("worker-result-present")
        if not worker.observations.get("oneMillionRequiresApproval"):
            raise AssertionError("phase7 must keep the 1M approval gate")
        checks.append("one-million-approval-gate")
        return _ok_reopen("phase7", "H4", checks=tuple(checks), provenance=provenance)
    except Exception as exc:
        return _failed_reopen(
            "phase7",
            "H4",
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase7(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    return _finalize_from_local(
        "phase7",
        "H4",
        worker,
        reopen,
        branch="awaiting-1m-approval",
        next_phase=None,
        decision="blocked",
    )


def run_phase8_worker_body(
    config: ProfilingConfig,
    workDir: Path,
    *,
    includeRemote: bool,
) -> PhaseWorkerResult:
    provenance = provenance_from_config(config)
    workDir.mkdir(parents=True, exist_ok=True)
    observations: dict[str, Any] = {
        "hypothesis": "H4",
        "status": "blocked",
        "message": "10M requires explicit post-1M approval",
        "includeRemote": includeRemote,
        "kind": "derived",
    }
    return PhaseWorkerResult(
        phase="phase8",
        hypothesis="H4",
        status="ok",
        checks=("post-1m-approval-gate",),
        observations=observations,
        provenance=provenance,
    )


def run_phase8_reopen_body(config: ProfilingConfig) -> PhaseReopenResult:
    provenance = provenance_from_config(config)
    checks: list[str] = []
    try:
        worker = PhaseWorkerResult.model_validate(
            get_json(config.phaseWorkerResultUri("phase8"))
        )
        if worker.status != "ok":
            raise RuntimeError(worker.error or "phase8 worker failed")
        checks.append("worker-result-present")
        return _ok_reopen("phase8", "H4", checks=tuple(checks), provenance=provenance)
    except Exception as exc:
        return _failed_reopen(
            "phase8",
            "H4",
            checks=tuple(checks),
            error=f"{type(exc).__name__}: {exc}",
            provenance=provenance,
        )


def finalize_phase8(
    worker: PhaseWorkerResult,
    reopen: PhaseReopenResult,
) -> PhaseFinalResult:
    return _finalize_from_local(
        "phase8",
        "H4",
        worker,
        reopen,
        branch="awaiting-post-1m-approval",
        next_phase=None,
        decision="blocked",
    )


PHASE_WORKERS = {
    "phase0": lambda config, work, remote: run_phase0_worker_body(
        config, work, includeRemoteInventory=remote
    ),
    "phase1": lambda config, work, remote: run_phase1_worker_body(
        config, work, includeRemote=remote
    ),
    "phase2": lambda config, work, remote: run_phase2_worker_body(
        config, work, includeRemote=remote
    ),
    "phase3": lambda config, work, remote: run_phase3_worker_body(
        config, work, includeRemote=remote
    ),
    "phase4": lambda config, work, remote: run_phase4_worker_body(
        config, work, includeRemote=remote
    ),
    "phase5": lambda config, work, remote: run_phase5_worker_body(
        config, work, includeRemote=remote
    ),
    "phase6": lambda config, work, remote: run_phase6_worker_body(
        config, work, includeRemote=remote
    ),
    "phase7": lambda config, work, remote: run_phase7_worker_body(
        config, work, includeRemote=remote
    ),
    "phase8": lambda config, work, remote: run_phase8_worker_body(
        config, work, includeRemote=remote
    ),
}
PHASE_REOPENERS = {
    "phase0": run_phase0_reopen_body,
    "phase1": run_phase1_reopen_body,
    "phase2": run_phase2_reopen_body,
    "phase3": run_phase3_reopen_body,
    "phase4": run_phase4_reopen_body,
    "phase5": run_phase5_reopen_body,
    "phase6": run_phase6_reopen_body,
    "phase7": run_phase7_reopen_body,
    "phase8": run_phase8_reopen_body,
}
PHASE_FINALIZERS = {
    "phase0": finalize_phase0,
    "phase1": finalize_phase1,
    "phase2": finalize_phase2,
    "phase3": finalize_phase3,
    "phase4": finalize_phase4,
    "phase5": finalize_phase5,
    "phase6": finalize_phase6,
    "phase7": finalize_phase7,
    "phase8": finalize_phase8,
}
