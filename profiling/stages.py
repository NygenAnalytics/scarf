import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scarf import DataStore, H5adReader, H5adToZarr, configure_output
from scarf.storage.budget import resolve_budget
from scarf.storage.sharding import write_counts_t
from scarf.storage.stores import open_store
from scarf.storage.types import as_zarr_array, as_zarr_group

from profiling.config import (
    StageName,
    StageResources,
    StorageLayout,
    WorkflowParameters,
)
from profiling.metrics import ResourceMeasurement, ResourceSampler, StageTimer
from profiling.r2 import storage_options

configure_output(progress=False, timestamps=True)

CHILD_MONITOR_INTERVAL_SECONDS = 30.0
CHILD_WARNING_SECONDS = 1_800.0
CHILD_STOP_GRACE_SECONDS = 30.0


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
    details: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _open_datastore(
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    initialize: bool,
) -> DataStore:
    options = storage_options(storeUri)
    arguments: dict[str, Any] = {
        "nthreads": resources.workers,
        "zarr_mode": "r+",
        "zarrProfile": ("cloud" if storeUri.startswith("s3://") else "fast_local"),
        "storage_options": options,
        "mem_budget": resources.scarfMemoryBudget,
    }
    if initialize:
        arguments.update(
            {
                "assay_types": {workflow.assayName: "RNA"},
                "default_assay": workflow.assayName,
                "min_features_per_cell": workflow.minFeaturesPerCell,
                "min_cells_per_feature": workflow.minCellsPerFeature,
            }
        )
    return DataStore(storeUri, **arguments)


def _close_h5ad_reader(reader: H5adReader) -> None:
    if hasattr(reader, "h5") and hasattr(reader.h5, "close"):
        reader.h5.close()


def _prepare_create_store(
    *,
    localH5adPath: Path,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    storageLayout: StorageLayout | None = None,
) -> tuple[H5adReader, H5adToZarr]:
    options = storage_options(storeUri)
    reader = H5adReader(
        str(localH5adPath),
        matrix_key="X",
        cell_attrs_key="obs",
        cell_ids_key="_index",
        feature_attrs_key="var",
        feature_ids_key="_index",
        feature_name_key="feature_name",
    )
    layout_kwargs: dict[str, Any] = {}
    if storageLayout is not None:
        if storageLayout.targetChunkBytes is not None:
            layout_kwargs["targetChunkBytes"] = storageLayout.targetChunkBytes
        if storageLayout.targetShardBytes is not None:
            layout_kwargs["targetShardBytes"] = storageLayout.targetShardBytes
    try:
        writer = H5adToZarr(
            reader,
            storeUri,
            assay_name=workflow.assayName,
            storage_options=options,
            mem_budget=resources.scarfMemoryBudget,
            nthreads=resources.workers,
            **layout_kwargs,
        )
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
) -> dict[str, Any]:
    worker_dir = (
        workDir if workDir is not None else Path(tempfile.mkdtemp(prefix=workDirPrefix))
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / "request.json"
    status_path = worker_dir / "status.json"
    status_path.unlink(missing_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "storeUri": storeUri,
                "workflow": workflow.model_dump(mode="json"),
                "resources": resources.model_dump(mode="json"),
                "statusPath": str(status_path),
            }
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        workerModule,
        "--request",
        str(request_path),
    ]
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
    return status


def _run_leiden_in_subprocess(
    *,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
) -> dict[str, Any]:
    return _run_worker_in_subprocess(
        stageLabel="runLeiden",
        workerModule="profiling.leiden_worker",
        storeUri=storeUri,
        workflow=workflow,
        resources=resources,
        workDir=workDir,
        workDirPrefix="scarf-leiden-",
    )


def _run_paris_in_subprocess(
    *,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
) -> dict[str, Any]:
    return _run_worker_in_subprocess(
        stageLabel="runClustering",
        workerModule="profiling.paris_worker",
        storeUri=storeUri,
        workflow=workflow,
        resources=resources,
        workDir=workDir,
        workDirPrefix="scarf-paris-",
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
) -> _CountsTWriteContext:
    budget = resolve_budget(
        memory=resources.scarfMemoryBudget,
        workers=resources.workers,
    )
    root = open_store(
        storeUri,
        mode="r+",
        storage_options=storage_options(storeUri),
    )
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
    featureMajorLayout: bool = False,
) -> Any:
    counts_t = write_counts_t(
        context.counts,
        context.group,
        profile="cloud" if storeUri.startswith("s3://") else "fast_local",
        resources=context.budget,
        feature_major_layout=featureMajorLayout,
    )
    if counts_t is None:
        raise RuntimeError(
            f"write_counts_t skipped for {assayName} (Zarr format < 3); "
            "cannot create countsT"
        )
    return counts_t


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
    countsT.attrs["complete"] = False

    expected_shape = (int(context.counts.shape[1]), int(context.counts.shape[0]))
    if tuple(countsT.shape) != expected_shape:
        raise RuntimeError(
            f"countsT shape {tuple(countsT.shape)} != expected {expected_shape}"
        )
    if np.dtype(countsT.dtype) != np.dtype(context.counts.dtype):
        raise RuntimeError(
            f"countsT dtype {countsT.dtype} != counts dtype {context.counts.dtype}"
        )

    rng = np.random.default_rng(seed)
    feat_chunk = max(1, int(countsT.chunks[0]))
    cell_chunk = max(1, int(countsT.chunks[1]))
    n_feats, n_cells = countsT.shape
    checks: list[dict[str, Any]] = []
    for _ in range(max(0, nCheckTiles)):
        feat_start = int(rng.integers(0, n_feats))
        feat_start = (feat_start // feat_chunk) * feat_chunk
        cell_start = int(rng.integers(0, n_cells))
        cell_start = (cell_start // cell_chunk) * cell_chunk
        feat_end = min(feat_start + feat_chunk, n_feats)
        cell_end = min(cell_start + cell_chunk, n_cells)
        got = np.asarray(countsT[feat_start:feat_end, cell_start:cell_end])
        expect = np.asarray(context.counts[cell_start:cell_end, feat_start:feat_end]).T
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

    countsT.attrs["complete"] = True
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


def run_stage(
    stage: StageName,
    *,
    nRows: int,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    localH5adPath: Path | None = None,
    storageLayout: StorageLayout | None = None,
    workDir: Path | None = None,
    sampleIntervalSeconds: float = 0.25,
    containerMemoryMb: int | None = None,
    containerCpuRequest: float | None = None,
    containerCpuLimit: float | None = None,
    resetCgroupPeak: bool = True,
    invalidateCache: bool = False,
) -> StageRunResult:
    timer = StageTimer()
    sampler = ResourceSampler(
        sampleIntervalSeconds=sampleIntervalSeconds,
        resetCgroupPeak=resetCgroupPeak,
    )
    error: str | None = None
    status = "ok"
    measurement: ResourceMeasurement | None = None
    details: dict[str, Any] | None = None
    worker_timings: dict[str, Any] | None = None

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
                            storageLayout=storageLayout,
                        )
                    with timer.operation():
                        assert writer is not None
                        writer.dump(batch_size=workflow.h5adBatchSize)
                finally:
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
                        )
                    with timer.operation():
                        assert counts_context is not None
                        counts_t = _write_counts_t(
                            counts_context,
                            storeUri=storeUri,
                            assayName=workflow.assayName,
                            featureMajorLayout=(
                                workflow.countsTLayout == "featureMajor"
                            ),
                        )
                    with timer.validationPersistence():
                        assert counts_context is not None
                        details = _validate_counts_t(
                            counts_context,
                            counts_t,
                            storeUri=storeUri,
                            assayName=workflow.assayName,
                            resources=resources,
                            nCheckTiles=3,
                            seed=0,
                        )
                finally:
                    counts_t = None
                    counts_context = None
            elif stage == "initializeStore":
                store: DataStore | None = None
                try:
                    with timer.operation():
                        store = _open_datastore(
                            storeUri, workflow, resources, initialize=True
                        )
                finally:
                    store = None
            elif stage == "reopenStore":
                store = None
                try:
                    with timer.operation():
                        store = _open_datastore(
                            storeUri, workflow, resources, initialize=False
                        )
                finally:
                    store = None
            elif stage == "runLeiden":
                with timer.operation():
                    worker_timings = _run_leiden_in_subprocess(
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        workDir=workDir,
                    )
            elif stage == "runClustering":
                with timer.operation():
                    worker_timings = _run_paris_in_subprocess(
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        workDir=workDir,
                    )
            else:
                store = None
                try:
                    with timer.inputSetup():
                        print(
                            f"[run_stage] ENTER open_datastore stage={stage} "
                            f"store={storeUri}",
                            flush=True,
                        )
                        store = _open_datastore(
                            storeUri, workflow, resources, initialize=False
                        )
                    with timer.operation():
                        assert store is not None
                        print(
                            f"[run_stage] datastore open; ENTER analysis stage={stage}",
                            flush=True,
                        )
                        _run_analysis(
                            stage,
                            store,
                            workflow,
                            resources,
                            invalidateCache=invalidateCache,
                        )
                        print(
                            f"[run_stage] analysis DONE stage={stage}",
                            flush=True,
                        )
                finally:
                    store = None
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            measurement = sampler.stop()

    timings = timer.result
    seconds = timings.measuredOperationSeconds
    input_setup_seconds = timings.inputSetupSeconds
    if worker_timings is not None:
        worker_setup = worker_timings.get("inputSetupSeconds")
        worker_operation = worker_timings.get("operationSeconds")
        worker_whole = worker_timings.get("wholeWorkerSeconds")
        if isinstance(worker_setup, int | float) and not isinstance(worker_setup, bool):
            input_setup_seconds = float(worker_setup)
        if isinstance(worker_operation, int | float) and not isinstance(
            worker_operation, bool
        ):
            seconds = float(worker_operation)
        if isinstance(worker_whole, int | float) and not isinstance(worker_whole, bool):
            details = {
                "subprocessSeconds": timings.measuredOperationSeconds,
                "workerWholeSeconds": float(worker_whole),
            }
    if stage == "createStore" and details is None:
        details = {"countsWriteSeconds": seconds}
    resource_summary = summarize_resource_measurement(measurement)
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
        **resource_summary,
    )


def _run_analysis(
    stage: StageName,
    store: DataStore,
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    invalidateCache: bool = False,
) -> None:
    if stage == "filterCells":
        store.auto_filter_cells(
            attrs=workflow.filterAttrs,
            min_p=workflow.filterMinQuantile,
            max_p=workflow.filterMaxQuantile,
            show_qc_plots=False,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "markHvgs":
        store.mark_hvgs(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            min_cells=workflow.hvgMinCells,
            top_n=workflow.topN,
            show_plot=False,
            hvg_key_name=workflow.hvgKey,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "runNormalization":
        store.run_normalization(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "runPca":
        store.run_pca(
            from_assay=workflow.assayName,
            dims=workflow.dims,
            local_cache=workflow.graphLocalCache,
            show_elbow_plot=False,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "buildEmbeddingInitialization":
        store.build_embedding_initialization(
            from_assay=workflow.assayName,
            n_centroids=workflow.nCentroids,
            rand_state=workflow.graphSeed,
            kmeans_sampling=workflow.kmeansSampling,
            kmeans_batch_size=workflow.kmeansBatchSize,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "buildAnnIndex":
        store.build_ann_index(
            from_assay=workflow.assayName,
            ann_efc=min(100, max(workflow.k * 3, 50)),
            ann_ef=min(100, max(workflow.k * 3, 50)),
            ann_m=min(max(48, int(workflow.dims * 1.5)), 64),
            ann_parallel=workflow.annParallel,
            rand_state=workflow.graphSeed,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "queryNeighbors":
        store.query_neighbors(
            from_assay=workflow.assayName,
            k=workflow.k,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "buildConnectivityMap":
        store.build_connectivity_map(
            from_assay=workflow.assayName,
            update_state=True,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "runUmap":
        store.run_umap(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            n_epochs=workflow.umapEpochs,
            random_seed=workflow.umapSeed,
            label=workflow.umapLabel,
            parallel=workflow.umapParallel,
            nthreads=resources.workers,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "runLeiden":
        raise AssertionError("runLeiden must execute in its child process")
    if stage == "findMarkers":
        store.run_marker_search(
            from_assay=workflow.assayName,
            group_key=workflow.resolvedMarkerGroupKey,
            cell_key=workflow.cellKey,
            feat_key=workflow.markerFeatureKey,
            gene_batch_size=workflow.markerGeneBatchSize,
            n_threads=resources.workers,
            skip_save=False,
            invalidate_cache=invalidateCache,
        )
        return
    if stage == "runClustering":
        raise AssertionError("runClustering must execute in its child process")
    raise ValueError(f"No analysis operation for {stage}")
