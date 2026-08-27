import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scarf import DataStore, H5adReader, H5adToZarr, configure_output
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.storage import ArtifactRef
from scarf.storage.budget import resolve_budget
from scarf.storage.selections import validate_stored_selection_integrity
from scarf.storage.stores import open_store
from scarf.storage.types import as_zarr_array, as_zarr_group

from profiling.config import (
    CountMatrixConfig,
    StageName,
    StageResources,
    StorageIoConfig,
    WorkflowParameters,
)
from profiling.metrics import ResourceMeasurement, ResourceSampler, StageTimer
from profiling.r2 import storage_options

configure_output(progress=False, timestamps=True)

CHILD_MONITOR_INTERVAL_SECONDS = 30.0
CHILD_WARNING_SECONDS = 1_800.0
CHILD_STOP_GRACE_SECONDS = 30.0

PROFILE_STAGE_INPUTS: dict[StageName, dict[str, tuple[StageName, str]]] = {
    "markHvgs": {"cells": ("filterCells", "cell_selection")},
    "importClusters": {"cells": ("filterCells", "cell_selection")},
    "runNormalization": {
        "cells": ("filterCells", "cell_selection"),
        "features": ("markHvgs", "feature_selection"),
    },
    "runPca": {"normalized": ("runNormalization", "normalized")},
    "buildEmbeddingInitialization": {"coordinates": ("runPca", "reduction")},
    "buildAnnIndex": {"coordinates": ("runPca", "reduction")},
    "queryNeighbors": {"ann_index": ("buildAnnIndex", "ann_index")},
    "buildConnectivityMap": {"neighbors": ("queryNeighbors", "neighbors")},
    "runUmap": {
        "graph": ("buildConnectivityMap", "connectivity_map"),
        "initialization": (
            "buildEmbeddingInitialization",
            "embedding_initialization",
        ),
    },
    "runLeiden": {"graph": ("buildConnectivityMap", "connectivity_map")},
}


def profile_stage_inputs(
    workflow: WorkflowParameters,
    stage: StageName,
) -> dict[str, tuple[StageName, str]]:
    """Return exact artifact dependencies for one profiling stage."""
    inputs = dict(PROFILE_STAGE_INPUTS.get(stage, {}))
    cluster_stage: StageName = (
        "importClusters" if workflow.clusterSourceUri is not None else "runLeiden"
    )
    if stage == "findMarkers":
        inputs["clusters"] = (cluster_stage, "cluster_labels")
    elif stage == "validateExperiment":
        inputs.update(
            {
                "pca": ("runPca", "reduction"),
                "clusters": (cluster_stage, "cluster_labels"),
                "markers": ("findMarkers", "marker_table"),
            }
        )
    return inputs


@dataclass(frozen=True, slots=True)
class StageRunResult:
    stage: StageName
    nRows: int
    status: str
    seconds: float | None
    peakRssBytes: int | None
    peakCgroupBytes: int | None
    modalMemoryMb: int
    scarfMemoryBudget: int
    storeUri: str
    error: str | None = None
    inputSetupSeconds: float | None = None
    validationPersistenceSeconds: float | None = None
    wholeFunctionSeconds: float | None = None
    modalCpuRequest: float | None = None
    modalCpuLimit: float | None = None
    rssBaselineBytes: int | None = None
    rssIncrementalPeakBytes: int | None = None
    rssAfterBytes: int | None = None
    cgroupCurrentBaselineBytes: int | None = None
    cgroupCurrentPeakBytes: int | None = None
    cgroupCurrentAfterBytes: int | None = None
    operationBaselineBytes: int | None = None
    operationIncrementalPeakBytes: int | None = None
    operationPeakSource: str | None = None
    cgroupPeakScope: str | None = None
    processCpuSeconds: float | None = None
    childCpuSeconds: float | None = None
    details: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _count_matrix_policy(config: CountMatrixConfig | None) -> Any:
    if config is None:
        return None
    from scarf.storage.count_matrix import CountMatrixPolicy

    return CountMatrixPolicy(unitBytes=config.unitBytes, chunkBytes=config.chunkBytes)


def _storage_io_policy(config: StorageIoConfig | None) -> Any:
    if config is None:
        return None
    from scarf.storage.io_policy import StorageIoPolicy

    return StorageIoPolicy(
        readWorkers=config.readWorkers,
        computeWorkers=config.computeWorkers,
        writeWorkers=config.writeWorkers,
    )


def _wrap_store_probe(
    storeUri: str, options: dict[str, Any] | None, storeProbe: Any
) -> Any:
    from scarf.storage.stores import make_store

    from profiling.recording_store import wrap_recording_store

    resolved = make_store(storeUri, storage_options=options)
    if isinstance(resolved, str):
        return storeUri
    return wrap_recording_store(resolved, probe=storeProbe)


def _open_datastore(
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    initialize: bool,
    storeProbe: Any | None = None,
    storageIo: StorageIoConfig | None = None,
) -> DataStore:
    options = storage_options(storeUri)
    location: Any = storeUri
    if storeProbe is not None:
        location = _wrap_store_probe(storeUri, options, storeProbe)
    arguments: dict[str, Any] = {
        "nthreads": resources.workers,
        "zarr_mode": "r+",
        "zarrProfile": ("cloud" if storeUri.startswith("s3://") else "fast_local"),
        "storage_options": options,
        "mem_budget": resources.scarfMemoryBudget,
        "storageIo": _storage_io_policy(storageIo),
    }
    if initialize:
        arguments.update(
            {
                "assay_types": {workflow.assayName: "RNA"},
                "default_assay": workflow.assayName,
                "min_features_per_cell": workflow.minFeaturesPerCell,
            }
        )
    return DataStore(location, **arguments)


def _reset_initialization_stats(
    storeUri: str,
    workflow: WorkflowParameters,
) -> None:
    from scarf.metadata import MetaData

    root = open_store(
        storeUri,
        mode="r+",
        storage_options=storage_options(storeUri),
    )
    cells = MetaData(as_zarr_group(root["cellData"], name="cellData"))
    for name in (
        f"{workflow.assayName}_nCounts",
        f"{workflow.assayName}_nFeatures",
        f"{workflow.assayName}_percentMito",
        f"{workflow.assayName}_percentRibo",
    ):
        if name in cells.columns:
            cells.drop(name)

    assay = as_zarr_group(root[workflow.assayName], name=workflow.assayName)
    features = MetaData(
        as_zarr_group(
            assay["featureData"],
            name=f"{workflow.assayName}/featureData",
        )
    )
    for name in ("nCells", "dropOuts"):
        if name in features.columns:
            features.drop(name)

    percent_features = assay.attrs.get("percentFeatures", {})
    if isinstance(percent_features, dict):
        reset_names = {
            f"{workflow.assayName}_percentMito",
            f"{workflow.assayName}_percentRibo",
        }
        assay.attrs["percentFeatures"] = {
            str(name): str(pattern)
            for name, pattern in percent_features.items()
            if str(name) not in reset_names
        }


def _close_h5ad_reader(reader: H5adReader) -> None:
    if hasattr(reader, "h5") and hasattr(reader.h5, "close"):
        reader.h5.close()


def _prepare_create_store(
    *,
    localH5adPath: Path,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    countMatrix: CountMatrixConfig | None = None,
    storageIo: StorageIoConfig | None = None,
    storeProbe: Any | None = None,
) -> tuple[H5adReader, H5adToZarr]:
    options = storage_options(storeUri)
    location: Any = storeUri
    if storeProbe is not None:
        location = _wrap_store_probe(storeUri, options, storeProbe)
    reader = H5adReader(
        str(localH5adPath),
        matrix_key="X",
        cell_attrs_key="obs",
        cell_ids_key="_index",
        feature_attrs_key="var",
        feature_ids_key="_index",
        feature_name_key="feature_name",
    )
    try:
        writer = H5adToZarr(
            reader,
            location,
            assay_name=workflow.assayName,
            storage_options=options,
            mem_budget=resources.scarfMemoryBudget,
            nthreads=resources.workers,
            profile=("cloud" if storeUri.startswith("s3://") else "fast_local"),
            policy=_count_matrix_policy(countMatrix),
            io=_storage_io_policy(storageIo),
        )
        writer._parallelWriteLocation = storeUri
    except BaseException:
        _close_h5ad_reader(reader)
        raise
    return reader, writer


def _monitor_child_process(
    process: subprocess.Popen[bytes],
    *,
    stageLabel: str,
    warningSeconds: float = CHILD_WARNING_SECONDS,
    pollSeconds: float = CHILD_MONITOR_INTERVAL_SECONDS,
) -> int:
    if warningSeconds <= 0:
        raise ValueError("warningSeconds must be positive")
    if pollSeconds <= 0:
        raise ValueError("pollSeconds must be positive")

    started = time.monotonic()
    warned = False
    while True:
        try:
            return process.wait(timeout=pollSeconds)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(
                f"[{stageLabel}] child still running pid={process.pid} "
                f"elapsedSeconds={elapsed:.0f}",
                flush=True,
            )
            if not warned and elapsed >= warningSeconds:
                print(
                    f"[{stageLabel}] WARNING child exceeded "
                    f"{warningSeconds:.0f}s; continuing",
                    flush=True,
                )
                warned = True


def _stop_child_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=CHILD_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _read_worker_status(statusPath: Path, *, workerName: str) -> dict[str, Any] | None:
    if not statusPath.is_file():
        return None
    payload = json.loads(statusPath.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{workerName} status must be a JSON object")
    return payload


def _run_worker_in_subprocess(
    *,
    stageLabel: str,
    workerModule: str,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
    workDirPrefix: str,
    invalidateCache: bool = False,
    requestInputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_dir = (
        workDir if workDir is not None else Path(tempfile.mkdtemp(prefix=workDirPrefix))
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / "request.json"
    status_path = worker_dir / "status.json"
    status_path.unlink(missing_ok=True)
    request: dict[str, Any] = {
        "storeUri": storeUri,
        "workflow": workflow.model_dump(mode="json"),
        "resources": resources.model_dump(mode="json"),
        "statusPath": str(status_path),
        "invalidateCache": invalidateCache,
        "inputs": requestInputs or {},
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        workerModule,
        "--request",
        str(request_path),
    ]
    from profiling.metrics import child_cpu_seconds as read_child_cpu_seconds

    cpu_before = read_child_cpu_seconds()
    process = subprocess.Popen(command)
    print(
        f"[{stageLabel}] child started pid={process.pid} module={workerModule} "
        f"warningSeconds={CHILD_WARNING_SECONDS:.0f}",
        flush=True,
    )
    try:
        return_code = _monitor_child_process(process, stageLabel=stageLabel)
    except BaseException:
        _stop_child_process(process)
        raise

    child_cpu = max(0.0, read_child_cpu_seconds() - cpu_before)
    status = _read_worker_status(status_path, workerName=f"{stageLabel} worker")
    if return_code != 0:
        detail = status.get("error") if status is not None else None
        suffix = f": {detail}" if isinstance(detail, str) else ""
        raise RuntimeError(
            f"{stageLabel} worker exited with code {return_code}{suffix}"
        )
    if status is None or status.get("status") != "ok":
        raise RuntimeError(f"{stageLabel} worker exited without a successful status")
    print(f"[{stageLabel}] child completed pid={process.pid}", flush=True)
    if "childCpuSeconds" not in status:
        status["childCpuSeconds"] = child_cpu
    return status


def _run_leiden_in_subprocess(
    *,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
    graph: ArtifactRef,
    invalidateCache: bool = False,
) -> dict[str, Any]:
    return _run_worker_in_subprocess(
        stageLabel="runLeiden",
        workerModule="profiling.leiden_worker",
        storeUri=storeUri,
        workflow=workflow,
        resources=resources,
        workDir=workDir,
        workDirPrefix="scarf-leiden-",
        invalidateCache=invalidateCache,
        requestInputs={"graph": graph.to_dict()},
    )


def _peak_cgroup_bytes(measurement: ResourceMeasurement | None) -> int | None:
    if measurement is None:
        return None
    if measurement.operationPeakSource in {"cgroupMemoryCurrent", "cgroupMemoryPeak"}:
        return measurement.operationPeakBytes
    return measurement.cgroupMemoryCurrentPeakBytes


def summarize_resource_measurement(
    measurement: ResourceMeasurement | None,
) -> dict[str, Any]:
    return {
        "peakRssBytes": (
            measurement.processTreeRssPeakBytes if measurement is not None else None
        ),
        "peakCgroupBytes": _peak_cgroup_bytes(measurement),
        "rssBaselineBytes": (
            measurement.processTreeRssBaselineBytes if measurement is not None else None
        ),
        "rssIncrementalPeakBytes": (
            measurement.processTreeRssIncrementalPeakBytes
            if measurement is not None
            else None
        ),
        "rssAfterBytes": (
            measurement.processTreeRssAfterBytes if measurement is not None else None
        ),
        "cgroupCurrentBaselineBytes": (
            measurement.cgroupMemoryCurrentBaselineBytes
            if measurement is not None
            else None
        ),
        "cgroupCurrentPeakBytes": (
            measurement.cgroupMemoryCurrentPeakBytes
            if measurement is not None
            else None
        ),
        "cgroupCurrentAfterBytes": (
            measurement.cgroupMemoryCurrentAfterBytes
            if measurement is not None
            else None
        ),
        "operationBaselineBytes": (
            measurement.operationBaselineBytes if measurement is not None else None
        ),
        "operationIncrementalPeakBytes": (
            measurement.operationIncrementalPeakBytes
            if measurement is not None
            else None
        ),
        "operationPeakSource": (
            measurement.operationPeakSource if measurement is not None else None
        ),
        "cgroupPeakScope": (
            measurement.cgroupMemoryPeakScope if measurement is not None else None
        ),
    }


@dataclass(frozen=True, slots=True)
class _CountsTWriteContext:
    counts: Any
    group: Any
    budget: Any
    beforeComplete: Any


def _prepare_counts_t_write(
    *,
    storeUri: str,
    assayName: str,
    resources: StageResources,
    storeProbe: Any | None = None,
) -> _CountsTWriteContext:
    budget = resolve_budget(
        memory=resources.scarfMemoryBudget,
        workers=resources.workers,
    )
    options = storage_options(storeUri)
    location: Any = storeUri
    if storeProbe is not None:
        location = _wrap_store_probe(storeUri, options, storeProbe)
    root = open_store(location, mode="r+", storage_options=options)
    group = as_zarr_group(root[assayName], name=assayName)
    counts = as_zarr_array(group["counts"], name=f"{assayName}/counts")
    before_complete = None
    if "countsT" in group:
        before_complete = group["countsT"].attrs.get("complete")
    return _CountsTWriteContext(
        counts=counts,
        group=group,
        budget=budget,
        beforeComplete=before_complete,
    )


def _write_counts_t(
    context: _CountsTWriteContext,
    *,
    storeUri: str,
    assayName: str,
    countMatrix: CountMatrixConfig | None = None,
    storageIo: StorageIoConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    from scarf.storage.count_matrix import plan_count_matrix_pair
    from scarf.storage.sharding import write_counts_t

    profile = "cloud" if storeUri.startswith("s3://") else "fast_local"
    policy = _count_matrix_policy(countMatrix)
    pair_kwargs: dict[str, Any] = {"profile": profile}
    if policy is not None:
        pair_kwargs["policy"] = policy
    pair = plan_count_matrix_pair(
        int(context.counts.shape[0]),
        int(context.counts.shape[1]),
        context.counts.dtype,
        **pair_kwargs,
    )
    writer_metrics: dict[str, Any] = {}
    counts_t = write_counts_t(
        context.counts,
        context.group,
        profile=profile,
        resources=context.budget,
        policy=policy,
        io=_storage_io_policy(storageIo),
        metrics=writer_metrics,
    )
    return counts_t, {
        "writer": "product",
        "fingerprint": pair.fingerprint,
        "sourceDecodeAmplification": pair.sourceDecodeAmplification,
        "countsTChunks": list(pair.countsT.chunks),
        "countsTShards": list(pair.countsT.shards or ()),
        "metrics": writer_metrics,
        "kind": "observed",
    }


def _validate_counts_t(
    context: _CountsTWriteContext,
    countsT: Any,
    *,
    storeUri: str,
    assayName: str,
    resources: StageResources,
    nCheckTiles: int,
    seed: int,
) -> dict[str, Any]:
    if countsT.attrs.get("complete") is not True:
        raise RuntimeError(
            f"countsT rewrite finished without complete=True at {storeUri}"
        )

    expected_shape = (int(context.counts.shape[1]), int(context.counts.shape[0]))
    if tuple(countsT.shape) != expected_shape:
        countsT.attrs["complete"] = False
        raise RuntimeError(
            f"countsT shape {tuple(countsT.shape)} != expected {expected_shape}"
        )
    if np.dtype(countsT.dtype) != np.dtype(context.counts.dtype):
        countsT.attrs["complete"] = False
        raise RuntimeError(
            f"countsT dtype {countsT.dtype} != counts dtype {context.counts.dtype}"
        )

    from scarf.storage.sharding import is_readable_counts_t_layout
    from scarf.storage.types import array_metadata_shards

    shards = array_metadata_shards(countsT)
    layout_arguments = {
        "shape": tuple(int(v) for v in countsT.shape),
        "chunks": tuple(int(v) for v in countsT.chunks),
        "shards": None if shards is None else tuple(int(v) for v in shards),
        "dtype": countsT.dtype,
    }
    valid_layout = shards is not None and is_readable_counts_t_layout(
        **layout_arguments
    )
    if not valid_layout:
        countsT.attrs["complete"] = False
        raise RuntimeError(
            f"countsT at {storeUri} does not match the requested writer layout"
        )

    rng = np.random.default_rng(seed)
    feat_chunk = max(1, int(countsT.chunks[0]))
    cell_chunk = max(1, int(countsT.chunks[1]))
    n_feats, n_cells = countsT.shape
    checks: list[dict[str, Any]] = []
    try:
        for _ in range(max(0, nCheckTiles)):
            feat_start = int(rng.integers(0, n_feats))
            feat_start = (feat_start // feat_chunk) * feat_chunk
            cell_start = int(rng.integers(0, n_cells))
            cell_start = (cell_start // cell_chunk) * cell_chunk
            feat_end = min(feat_start + feat_chunk, n_feats)
            cell_end = min(cell_start + cell_chunk, n_cells)
            got = np.asarray(countsT[feat_start:feat_end, cell_start:cell_end])
            expect = np.asarray(
                context.counts[cell_start:cell_end, feat_start:feat_end]
            ).T
            if got.shape != expect.shape or not np.array_equal(got, expect):
                raise RuntimeError(
                    "countsT tile mismatch after rewrite "
                    f"feat=[{feat_start}:{feat_end}] cell=[{cell_start}:{cell_end}]"
                )
            checks.append(
                {
                    "featStart": feat_start,
                    "featEnd": feat_end,
                    "cellStart": cell_start,
                    "cellEnd": cell_end,
                }
            )
    except Exception:
        countsT.attrs["complete"] = False
        raise

    return {
        "assayName": assayName,
        "beforeComplete": context.beforeComplete,
        "complete": True,
        "shape": list(countsT.shape),
        "chunks": list(countsT.chunks),
        "dtype": str(countsT.dtype),
        "workers": resources.workers,
        "checkedTiles": checks,
    }


def install_stage_zarr_runtime(resources: StageResources) -> None:
    """Install the process Zarr runtime once from the first stage budget."""
    from scarf.storage.async_execution import (
        configure_zarr_runtime,
        resolve_execution_plan,
        zarr_runtime_installed,
    )

    if zarr_runtime_installed():
        return
    runtime = resolve_execution_plan(
        resolve_budget(
            memory=resources.scarfMemoryBudget,
            workers=resources.workers,
        )
    )
    configure_zarr_runtime(
        codecWorkers=runtime.codecWorkerLimit,
        asyncConcurrency=runtime.zarrAsyncConcurrency,
    )


def run_stage(
    stage: StageName,
    *,
    nRows: int,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    localH5adPath: Path | None = None,
    countMatrix: CountMatrixConfig | None = None,
    storageIo: StorageIoConfig | None = None,
    workDir: Path | None = None,
    sampleIntervalSeconds: float = 0.25,
    containerMemoryMb: int | None = None,
    containerCpuRequest: float | None = None,
    containerCpuLimit: float | None = None,
    resetCgroupPeak: bool = True,
    invalidateCache: bool = False,
    recordStoreOperations: bool = True,
    clientProvenance: dict[str, Any] | None = None,
    inputRefs: dict[str, ArtifactRef] | None = None,
    session: dict[str, Any] | None = None,
) -> StageRunResult:
    install_stage_zarr_runtime(resources)
    timer = StageTimer()
    cpu_started = time.process_time()
    sampler = ResourceSampler(
        sampleIntervalSeconds=sampleIntervalSeconds,
        resetCgroupPeak=resetCgroupPeak,
    )
    error: str | None = None
    status = "ok"
    measurement: ResourceMeasurement | None = None
    details: dict[str, Any] | None = None
    worker_timings: dict[str, Any] | None = None
    store_probe: Any | None = None
    if recordStoreOperations:
        from profiling.recording_store import StoreProbe

        store_probe = StoreProbe(countOnly=True)
    collected_reports: list[Any] = []

    from scarf.storage.execution import execution_report_scope

    _shared_store_stages = {
        "filterCells",
        "markHvgs",
        "runNormalization",
        "runPca",
        "buildEmbeddingInitialization",
        "buildAnnIndex",
        "queryNeighbors",
        "buildConnectivityMap",
        "runUmap",
        "findMarkers",
        "importClusters",
        "validateExperiment",
    }

    def _keep_store(opened: DataStore | None) -> None:
        if session is not None and opened is not None:
            session["store"] = opened

    resolved_input_refs = dict(inputRefs or {})
    if session is not None:
        session_refs = session.get("artifactRefs")
        if isinstance(session_refs, dict):
            for input_name, (source_stage, _kind) in profile_stage_inputs(
                workflow, stage
            ).items():
                candidate = session_refs.get(source_stage)
                if isinstance(candidate, ArtifactRef):
                    resolved_input_refs.setdefault(input_name, candidate)

    from profiling.metrics import child_cpu_seconds as read_child_cpu_seconds

    child_cpu_before = read_child_cpu_seconds()
    report_scope = execution_report_scope()
    collected_reports = report_scope.__enter__()
    try:
        with timer:
            sampler.start()
            try:
                if stage == "createStore":
                    if localH5adPath is None:
                        raise ValueError("createStore requires localH5adPath")
                    reader: H5adReader | None = None
                    writer: H5adToZarr | None = None
                    try:
                        with timer.inputSetup():
                            reader, writer = _prepare_create_store(
                                localH5adPath=localH5adPath,
                                storeUri=storeUri,
                                workflow=workflow,
                                resources=resources,
                                countMatrix=countMatrix,
                                storageIo=storageIo,
                                storeProbe=store_probe,
                            )
                        with timer.operation():
                            assert writer is not None
                            # Keep createStore = counts only. writeCountsT owns
                            # paired countsT so the two stages stay measurable.
                            writer._write_counts(batch_size=workflow.h5adBatchSize)
                    finally:
                        if writer is not None:
                            details = {
                                "h5adProducerWorkers": getattr(
                                    writer,
                                    "_lastImportProducerCount",
                                    None,
                                ),
                                "h5adWriteWorkers": getattr(
                                    writer,
                                    "_lastImportWriteWorkers",
                                    None,
                                ),
                                "h5adWorkersPerProcess": getattr(
                                    writer,
                                    "_lastImportWorkersPerProcess",
                                    None,
                                ),
                            }
                        writer = None
                        if reader is not None:
                            _close_h5ad_reader(reader)
                            reader = None
                elif stage == "writeCountsT":
                    counts_context: _CountsTWriteContext | None = None
                    counts_t: Any = None
                    try:
                        with timer.inputSetup():
                            counts_context = _prepare_counts_t_write(
                                storeUri=storeUri,
                                assayName=workflow.assayName,
                                resources=resources,
                                storeProbe=store_probe,
                            )
                        with timer.operation():
                            assert counts_context is not None
                            counts_t, write_details = _write_counts_t(
                                counts_context,
                                storeUri=storeUri,
                                assayName=workflow.assayName,
                                countMatrix=countMatrix,
                                storageIo=storageIo,
                            )
                        with timer.validationPersistence():
                            assert counts_context is not None
                            details = {
                                **write_details,
                                **_validate_counts_t(
                                    counts_context,
                                    counts_t,
                                    storeUri=storeUri,
                                    assayName=workflow.assayName,
                                    resources=resources,
                                    nCheckTiles=3,
                                    seed=0,
                                ),
                            }
                    finally:
                        counts_t = None
                        counts_context = None
                elif stage == "initializeStore":
                    store: DataStore | None = None
                    try:
                        if invalidateCache:
                            _reset_initialization_stats(storeUri, workflow)
                        with timer.operation():
                            store = _open_datastore(
                                storeUri,
                                workflow,
                                resources,
                                initialize=True,
                                storeProbe=store_probe,
                                storageIo=storageIo,
                            )
                            _keep_store(store)
                    finally:
                        if session is None:
                            store = None
                elif stage == "reopenStore":
                    store = None
                    try:
                        with timer.operation():
                            store = _open_datastore(
                                storeUri,
                                workflow,
                                resources,
                                initialize=False,
                                storeProbe=store_probe,
                                storageIo=storageIo,
                            )
                    finally:
                        if session is None:
                            store = None
                elif stage == "runLeiden":
                    graph = _profile_input(
                        resolved_input_refs,
                        name="graph",
                        kind="connectivity_map",
                        assay=workflow.assayName,
                    )
                    with timer.operation():
                        worker_timings = _run_leiden_in_subprocess(
                            storeUri=storeUri,
                            workflow=workflow,
                            resources=resources,
                            workDir=workDir,
                            graph=graph,
                            invalidateCache=invalidateCache,
                        )
                    raw_artifact = worker_timings.get("artifact")
                    if not isinstance(raw_artifact, dict):
                        raise RuntimeError(
                            "runLeiden worker did not return its cluster artifact"
                        )
                    cluster_ref = ArtifactRef.from_dict(raw_artifact)
                    if (
                        cluster_ref.scope != "assay"
                        or cluster_ref.assay != workflow.assayName
                        or cluster_ref.kind != "cluster_labels"
                    ):
                        raise RuntimeError(
                            "runLeiden worker returned an incompatible artifact"
                        )
                    details = {"artifact": cluster_ref.to_dict()}
                    if session is not None:
                        session.setdefault("artifactRefs", {})[stage] = cluster_ref
                else:
                    store = None
                    reused = (
                        session.get("store")
                        if session is not None and stage in _shared_store_stages
                        else None
                    )
                    try:
                        if reused is not None:
                            store = reused
                        else:
                            with timer.inputSetup():
                                print(
                                    f"[run_stage] ENTER open_datastore stage={stage} "
                                    f"store={storeUri}",
                                    flush=True,
                                )
                                store = _open_datastore(
                                    storeUri,
                                    workflow,
                                    resources,
                                    initialize=False,
                                    storeProbe=store_probe,
                                    storageIo=storageIo,
                                )
                                _keep_store(store)
                        with timer.operation():
                            assert store is not None
                            print(
                                f"[run_stage] datastore open; ENTER analysis stage={stage}",
                                flush=True,
                            )
                            analysis_details = _run_analysis(
                                stage,
                                store,
                                workflow,
                                resources,
                                invalidateCache=invalidateCache,
                                inputRefs=resolved_input_refs,
                            )
                            if analysis_details:
                                details = {
                                    **(details or {}),
                                    **analysis_details,
                                }
                                if session is not None:
                                    artifact = analysis_details.get("artifact")
                                    if isinstance(artifact, dict):
                                        session.setdefault("artifactRefs", {})[
                                            stage
                                        ] = ArtifactRef.from_dict(artifact)
                            print(
                                f"[run_stage] analysis DONE stage={stage}",
                                flush=True,
                            )
                    finally:
                        if session is None:
                            store = None
            except Exception as exc:
                status = "error"
                error = (
                    "".join(traceback.format_exception(exc))
                    if isinstance(exc, BaseExceptionGroup)
                    else f"{type(exc).__name__}: {exc}"
                )
            finally:
                measurement = sampler.stop()
    finally:
        report_scope.__exit__(None, None, None)

    timings = timer.result
    seconds = timings.measuredOperationSeconds
    input_setup_seconds = timings.inputSetupSeconds
    measured_child_cpu = max(0.0, read_child_cpu_seconds() - child_cpu_before)
    child_cpu_seconds: float | None = (
        measured_child_cpu if measured_child_cpu > 0 else None
    )
    if worker_timings is not None:
        worker_setup = worker_timings.get("inputSetupSeconds")
        worker_operation = worker_timings.get("operationSeconds")
        worker_whole = worker_timings.get("wholeWorkerSeconds")
        worker_child_cpu = worker_timings.get("childCpuSeconds")
        worker_process_cpu = worker_timings.get("processCpuSeconds")
        if isinstance(worker_setup, int | float) and not isinstance(worker_setup, bool):
            input_setup_seconds = float(worker_setup)
        if isinstance(worker_operation, int | float) and not isinstance(
            worker_operation, bool
        ):
            seconds = float(worker_operation)
        if isinstance(worker_child_cpu, int | float) and not isinstance(
            worker_child_cpu, bool
        ):
            child_cpu_seconds = float(worker_child_cpu)
        extra_details: dict[str, Any] = {}
        if isinstance(worker_whole, int | float) and not isinstance(worker_whole, bool):
            extra_details["subprocessSeconds"] = timings.measuredOperationSeconds
            extra_details["workerWholeSeconds"] = float(worker_whole)
        if isinstance(worker_process_cpu, int | float) and not isinstance(
            worker_process_cpu, bool
        ):
            extra_details["workerProcessCpuSeconds"] = float(worker_process_cpu)
        label_sha256 = worker_timings.get("labelSha256")
        if isinstance(label_sha256, str):
            extra_details["labelSha256"] = label_sha256
        cluster_count = worker_timings.get("clusterCount")
        if isinstance(cluster_count, int) and not isinstance(cluster_count, bool):
            extra_details["clusterCount"] = cluster_count
        if extra_details:
            details = {**(details or {}), **extra_details}
    if stage == "createStore":
        counts_t_present = False
        if status == "ok":
            root = open_store(
                storeUri,
                mode="r",
                storage_options=storage_options(storeUri),
            )
            assay = as_zarr_group(
                root[workflow.assayName],
                name=workflow.assayName,
            )
            counts_t_present = "countsT" in assay
        details = {
            "countsWriteSeconds": seconds,
            "countsOnly": True,
            "countsTPresent": counts_t_present,
            **(details or {}),
        }
    if store_probe is not None:
        details = {
            **(details or {}),
            "storeOperations": store_probe.to_json(),
        }
    if collected_reports:
        from scarf.storage.execution import execution_reports_by_kind

        details = {
            **(details or {}),
            "executionReports": execution_reports_by_kind(collected_reports),
        }
    resource_summary = summarize_resource_measurement(measurement)
    process_cpu_seconds = time.process_time() - cpu_started
    from profiling.provenance import collect_run_provenance

    return StageRunResult(
        stage=stage,
        nRows=nRows,
        status=status,
        seconds=seconds,
        modalMemoryMb=(
            resources.modalMemoryLimitMb
            if containerMemoryMb is None
            else containerMemoryMb
        ),
        modalCpuRequest=(
            resources.modalCpuRequest
            if containerCpuRequest is None
            else containerCpuRequest
        ),
        modalCpuLimit=(
            resources.modalCpuLimit if containerCpuLimit is None else containerCpuLimit
        ),
        scarfMemoryBudget=resources.scarfMemoryBudget,
        storeUri=storeUri,
        error=error,
        inputSetupSeconds=input_setup_seconds,
        validationPersistenceSeconds=timings.validationPersistenceSeconds,
        wholeFunctionSeconds=timings.wholeFunctionSeconds,
        details=details,
        processCpuSeconds=process_cpu_seconds,
        childCpuSeconds=child_cpu_seconds,
        provenance=collect_run_provenance(
            nonpreemptible=True,
            clientProvenance=clientProvenance,
        ),
        **resource_summary,
    )


def _feature_consume_details(
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    unitKind: str | None = None,
) -> dict[str, Any]:
    from scarf.storage.execution import (
        last_execution_report,
        recorded_execution_reports,
    )

    payload: dict[str, Any] = {
        "workers": resources.workers,
        "scarfMemoryBudget": resources.scarfMemoryBudget,
        "kind": "observed",
    }
    selected = None
    reports = recorded_execution_reports()
    if unitKind is not None:
        matching = [report for report in reports if report.unitKind == unitKind]
        if matching:
            selected = matching[-1]
    if selected is None:
        selected = last_execution_report()
    if selected is not None:
        payload.update(selected.as_metrics())
    return payload


def validate_cluster_source_identity(
    *,
    sourceIds: np.ndarray,
    targetIds: np.ndarray,
    sourceActive: np.ndarray,
    targetActive: np.ndarray,
    labels: np.ndarray,
) -> list[str]:
    """Validate compact cluster labels against identical stored selections."""
    if sourceIds.shape != targetIds.shape:
        raise ValueError(
            "cluster source row count does not match the target store; "
            "do not substitute labels by row count"
        )
    if not np.array_equal(sourceIds.astype(str), targetIds.astype(str)):
        raise ValueError(
            "cluster source cell ids are not identical in order; "
            "reordered or mismatched identities are rejected"
        )
    if int(np.unique(sourceIds.astype(str)).shape[0]) != int(sourceIds.shape[0]):
        raise ValueError("cluster source cell ids are not unique")
    if not np.array_equal(
        np.asarray(sourceActive).astype(bool),
        np.asarray(targetActive).astype(bool),
    ):
        raise ValueError("cluster source active-cell mask does not match the target")
    selected_count = int(np.asarray(sourceActive, dtype=bool).sum())
    if labels.shape != (selected_count,):
        raise ValueError("cluster source labels do not cover every selected cell")
    if any(str(value) in {"", "nan", "None"} for value in labels):
        raise ValueError("cluster source has missing labels on selected cells")
    groups = sorted({str(value) for value in labels})
    if len(groups) < 2:
        raise ValueError("cluster source must contain at least two groups")
    return groups


def validate_experiment_branches(
    *,
    pcaComplete: bool,
    importedClusterComplete: bool,
    markerComplete: bool,
) -> dict[str, bool]:
    """Validate the PCA branch and the imported-marker branch separately."""
    if not pcaComplete:
        raise ValueError(
            "validateExperiment: PCA branch is missing a reduction artifact"
        )
    if not importedClusterComplete:
        raise ValueError("validateExperiment: imported cluster artifact is missing")
    if not markerComplete:
        raise ValueError("validateExperiment: marker branch is missing a marker_table")
    return {"pcaBranch": True, "markerBranch": True}


def _ordered_id_digest(values: np.ndarray) -> str:
    import hashlib

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


def _import_cluster_labels(
    store: DataStore,
    workflow: WorkflowParameters,
    *,
    cell_selection: ArtifactRef,
) -> dict[str, Any]:
    source_uri = workflow.clusterSourceUri
    if source_uri is None or not source_uri.strip():
        raise ValueError("importClusters requires workflow.clusterSourceUri")
    source_artifact_id = workflow.clusterSourceArtifactId
    if source_artifact_id is None:
        raise ValueError("importClusters requires workflow.clusterSourceArtifactId")
    options = storage_options(source_uri)
    source_store = DataStore(
        source_uri,
        default_assay=workflow.assayName,
        zarr_mode="r",
        zarrProfile=("cloud" if source_uri.startswith("s3://") else "fast_local"),
        storage_options=options,
    )
    source_ref = ArtifactRef(
        scope="assay",
        assay=workflow.assayName,
        kind="cluster_labels",
        artifact_id=source_artifact_id,
    )
    source_status = source_store.inspect_artifact(source_ref)
    if not source_status.exists:
        raise ValueError(
            f"cluster source artifact does not exist: {source_status.path}"
        )
    if not source_status.complete:
        raise ValueError(f"cluster source artifact is incomplete: {source_status.path}")
    raw_source_selection = (source_status.inputs or {}).get("cell_selection")
    if not isinstance(raw_source_selection, dict):
        raise ValueError("cluster source artifact has no cell-selection input")
    try:
        source_selection_ref = ArtifactRef.from_dict(raw_source_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "cluster source artifact cell-selection input is malformed"
        ) from exc
    source_selection = validate_stored_selection_integrity(
        source_store.zw,
        source_selection_ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    target_selection = validate_stored_selection_integrity(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    source_group = source_store.load_artifact(source_ref)
    if "values" not in source_group:
        raise ValueError("cluster source artifact has no values")
    labels = np.asarray(as_zarr_array(source_group["values"], name="values")[:])
    source_ids = np.asarray(source_selection.row_ids[:])
    target_ids = np.asarray(target_selection.row_ids[:])
    source_active = np.asarray(source_selection.values[:], dtype=bool)
    target_active = np.asarray(target_selection.values[:], dtype=bool)
    groups = validate_cluster_source_identity(
        sourceIds=source_ids,
        targetIds=target_ids,
        sourceActive=source_active,
        targetActive=target_active,
        labels=labels,
    )
    dtype_kind = None if labels.dtype.kind in {"O", "S", "U"} else labels.dtype.kind
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay=workflow.assayName,
        kind="cluster_labels",
        operation="import_cluster_labels",
        parameters={
            "source_uri": source_uri,
            "source_artifact_id": source_ref.artifact_id,
        },
        inputs={
            "source_row_ids_fingerprint": _ordered_id_digest(source_ids),
            "source_labels_fingerprint": _ordered_id_digest(
                np.asarray(labels, dtype=object)
            ),
            "source_cell_selection_id": source_selection_ref.artifact_id,
        },
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (labels.shape, dtype_kind)},
    )
    write_cell_data_artifact(
        store.zw,
        planned,
        {"values": labels},
    )
    return {
        "artifact": planned.ref.to_dict(),
        "sourceUri": source_uri,
        "sourceArtifact": source_ref.to_dict(),
        "rowSelectionFingerprint": _ordered_id_digest(source_ids),
        "labelFingerprint": _ordered_id_digest(np.asarray(labels, dtype=object)),
        "groupCount": len(groups),
        "activeCells": target_selection.selected_count,
        "kind": "observed",
    }


def _validate_experiment(
    store: DataStore,
    workflow: WorkflowParameters,
    *,
    pca: ArtifactRef,
    clusters: ArtifactRef,
    markers: ArtifactRef,
) -> dict[str, Any]:
    pca_status = store.inspect_artifact(pca)
    cluster_status = store.inspect_artifact(clusters)
    marker_status = store.inspect_artifact(markers)
    pca_complete = pca_status.complete
    imported = clusters.kind == "cluster_labels" and cluster_status.complete
    marker_complete = marker_status.complete
    validate_experiment_branches(
        pcaComplete=pca_complete,
        importedClusterComplete=imported,
        markerComplete=marker_complete,
    )
    return {
        "pcaBranch": {
            "artifactId": pca.artifact_id,
            "complete": pca_status.complete,
        },
        "markerBranch": {
            "artifactId": markers.artifact_id,
            "complete": marker_status.complete,
            "clustersArtifactId": clusters.artifact_id,
        },
        "kind": "observed",
    }


def _profile_input(
    inputs: dict[str, ArtifactRef],
    *,
    name: str,
    kind: str,
    assay: str,
) -> ArtifactRef:
    ref = inputs.get(name)
    if ref is None:
        raise ValueError(f"Profiling stage requires explicit {name!r} input ({kind})")
    expected_scope = "datastore" if kind == "cell_selection" else "assay"
    expected_assay = None if expected_scope == "datastore" else assay
    if ref.scope != expected_scope or ref.assay != expected_assay or ref.kind != kind:
        raise ValueError(
            f"Profiling input {name!r} must be a {expected_scope}-scoped "
            f"{kind} artifact for {expected_assay!r}"
        )
    return ref


def _run_analysis(
    stage: StageName,
    store: DataStore,
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    invalidateCache: bool = False,
    inputRefs: dict[str, ArtifactRef] | None = None,
) -> dict[str, Any] | None:
    inputs = inputRefs or {}
    if stage == "filterCells":
        ref = store.auto_filter_cells(
            attrs=workflow.filterAttrs,
            min_p=workflow.filterMinQuantile,
            max_p=workflow.filterMaxQuantile,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "markHvgs":
        cell_selection = _profile_input(
            inputs,
            name="cells",
            kind="cell_selection",
            assay=workflow.assayName,
        )
        ref = store.select_hvgs(
            cell_selection,
            from_assay=workflow.assayName,
            min_cells=workflow.hvgMinCells,
            top_n=workflow.topN,
            show_plot=False,
            invalidate_cache=invalidateCache,
        )
        return {
            "artifact": ref.to_dict(),
            "consume": _feature_consume_details(
                workflow,
                resources,
                unitKind="countsTCellBand",
            ),
        }
    if stage == "runNormalization":
        feature_selection = _profile_input(
            inputs,
            name="features",
            kind="feature_selection",
            assay=workflow.assayName,
        )
        cell_selection = _profile_input(
            inputs,
            name="cells",
            kind="cell_selection",
            assay=workflow.assayName,
        )
        ref = store.run_normalization(
            cell_selection,
            feature_selection,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "runPca":
        normalized = _profile_input(
            inputs,
            name="normalized",
            kind="normalized",
            assay=workflow.assayName,
        )
        ref = store.run_pca(
            normalized,
            dims=workflow.dims,
            local_cache=workflow.graphLocalCache,
            show_elbow_plot=False,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "buildEmbeddingInitialization":
        reduction = _profile_input(
            inputs,
            name="coordinates",
            kind="reduction",
            assay=workflow.assayName,
        )
        ref = store.build_embedding_initialization(
            reduction,
            n_centroids=workflow.nCentroids,
            rand_state=workflow.graphSeed,
            kmeans_sampling=workflow.kmeansSampling,
            kmeans_batch_size=workflow.kmeansBatchSize,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "buildAnnIndex":
        reduction = _profile_input(
            inputs,
            name="coordinates",
            kind="reduction",
            assay=workflow.assayName,
        )
        ref = store.build_ann_index(
            reduction,
            ann_efc=min(100, max(workflow.k * 3, 50)),
            ann_ef=min(100, max(workflow.k * 3, 50)),
            ann_m=min(max(48, int(workflow.dims * 1.5)), 64),
            ann_parallel=workflow.annParallel,
            rand_state=workflow.graphSeed,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "queryNeighbors":
        ann_index = _profile_input(
            inputs,
            name="ann_index",
            kind="ann_index",
            assay=workflow.assayName,
        )
        ref = store.query_neighbors(
            ann_index,
            k=workflow.k,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "buildConnectivityMap":
        neighbors = _profile_input(
            inputs,
            name="neighbors",
            kind="neighbors",
            assay=workflow.assayName,
        )
        ref = store.build_connectivity_map(
            neighbors,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "runUmap":
        graph = _profile_input(
            inputs,
            name="graph",
            kind="connectivity_map",
            assay=workflow.assayName,
        )
        initialization = _profile_input(
            inputs,
            name="initialization",
            kind="embedding_initialization",
            assay=workflow.assayName,
        )
        ref = store.run_umap(
            graph,
            initialization,
            n_epochs=workflow.umapEpochs,
            random_seed=workflow.umapSeed,
            parallel=workflow.umapParallel,
            nthreads=resources.workers,
            invalidate_cache=invalidateCache,
        )
        return {"artifact": ref.to_dict()}
    if stage == "runLeiden":
        raise AssertionError("runLeiden must execute in its child process")
    if stage == "findMarkers":
        clusters = _profile_input(
            inputs,
            name="clusters",
            kind="cluster_labels",
            assay=workflow.assayName,
        )
        feature_selection = store.set_feature_selection(
            from_assay=workflow.assayName,
            mask=np.ones(
                store.get_assay(workflow.assayName).feats.N,
                dtype=bool,
            ),
            invalidate_cache=invalidateCache,
        )
        ref = store.run_marker_search(
            clusters,
            from_assay=workflow.assayName,
            features=feature_selection,
            nthreads=resources.workers,
            invalidate_cache=invalidateCache,
        )
        return {
            "artifact": ref.to_dict(),
            "consume": _feature_consume_details(
                workflow,
                resources,
                unitKind="countsTReadGroup",
            ),
        }
    if stage == "importClusters":
        cell_selection = _profile_input(
            inputs,
            name="cells",
            kind="cell_selection",
            assay=workflow.assayName,
        )
        return _import_cluster_labels(
            store,
            workflow,
            cell_selection=cell_selection,
        )
    if stage == "validateExperiment":
        pca = _profile_input(
            inputs,
            name="pca",
            kind="reduction",
            assay=workflow.assayName,
        )
        markers = _profile_input(
            inputs,
            name="markers",
            kind="marker_table",
            assay=workflow.assayName,
        )
        clusters = _profile_input(
            inputs,
            name="clusters",
            kind="cluster_labels",
            assay=workflow.assayName,
        )
        return _validate_experiment(
            store,
            workflow,
            pca=pca,
            clusters=clusters,
            markers=markers,
        )
    raise ValueError(f"No analysis operation for {stage}")
