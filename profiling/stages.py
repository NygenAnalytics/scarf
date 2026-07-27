import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scarf import DataStore, H5adReader, H5adToZarr, SubsetZarr
from scarf.storage.budget import resolve_budget
from scarf.storage.sharding import write_counts_t
from scarf.storage.stores import open_store
from scarf.storage.types import as_zarr_array, as_zarr_group
from scarf.writers import to_h5ad

from profiling.config import (
    StageName,
    StageResources,
    StorageLayout,
    WorkflowParameters,
)
from profiling.metrics import ResourceMeasurement, ResourceSampler, StageTimer
from profiling.r2 import storage_options

LEIDEN_MONITOR_INTERVAL_SECONDS = 30.0
LEIDEN_WARNING_SECONDS = 1_800.0
LEIDEN_STOP_GRACE_SECONDS = 30.0


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


def run_create_store(
    *,
    localH5adPath: Path,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    storageLayout: StorageLayout | None = None,
) -> None:
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
        writer.dump(batch_size=workflow.h5adBatchSize)
    finally:
        if hasattr(reader, "h5") and hasattr(reader.h5, "close"):
            reader.h5.close()


def _assay(store: DataStore, workflow: WorkflowParameters) -> Any:
    return getattr(store, workflow.assayName)


def _insert_synthetic_batches(store: DataStore, workflow: WorkflowParameters) -> None:
    n_active = int(store.cells.fetch(workflow.cellKey).sum())
    rng = np.random.default_rng(workflow.harmonyBatchSeed)
    batch_ids = rng.integers(0, workflow.harmonyNBatches, size=n_active)
    labels = np.array([f"b{int(i)}" for i in batch_ids], dtype=object)
    store.cells.insert(
        workflow.harmonyBatchColumn,
        labels,
        key=workflow.cellKey,
        overwrite=True,
    )


def _impute_feature_names(store: DataStore, workflow: WorkflowParameters) -> list[str]:
    assay = _assay(store, workflow)
    names = np.asarray(assay.feats.fetch_all("names"))
    keep = np.asarray(assay.feats.fetch_all("I"), dtype=bool)
    candidates = [str(name) for name, ok in zip(names, keep, strict=True) if ok]
    if not candidates:
        raise ValueError("No active features available for imputation")
    return candidates[: workflow.imputeGeneCount]


def _monitor_child_process(
    process: subprocess.Popen[bytes],
    *,
    stageLabel: str,
    warningSeconds: float = LEIDEN_WARNING_SECONDS,
    pollSeconds: float = LEIDEN_MONITOR_INTERVAL_SECONDS,
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


def _monitor_leiden_process(
    process: subprocess.Popen[bytes],
    *,
    warningSeconds: float = LEIDEN_WARNING_SECONDS,
    pollSeconds: float = LEIDEN_MONITOR_INTERVAL_SECONDS,
) -> int:
    return _monitor_child_process(
        process,
        stageLabel="runLeiden",
        warningSeconds=warningSeconds,
        pollSeconds=pollSeconds,
    )


def _stop_child_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=LEIDEN_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stop_leiden_process(process: subprocess.Popen[bytes]) -> None:
    _stop_child_process(process)


def _read_worker_status(statusPath: Path, *, workerName: str) -> dict[str, Any] | None:
    if not statusPath.is_file():
        return None
    payload = json.loads(statusPath.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{workerName} status must be a JSON object")
    return payload


def _read_leiden_worker_status(statusPath: Path) -> dict[str, Any] | None:
    return _read_worker_status(statusPath, workerName="Leiden worker")


def _run_worker_in_subprocess(
    *,
    stageLabel: str,
    workerModule: str,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
    workDirPrefix: str,
) -> None:
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
        f"warningSeconds={LEIDEN_WARNING_SECONDS:.0f}",
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


def _run_leiden_in_subprocess(
    *,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    workDir: Path | None,
) -> None:
    _run_worker_in_subprocess(
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
) -> None:
    _run_worker_in_subprocess(
        stageLabel="runClustering",
        workerModule="profiling.paris_worker",
        storeUri=storeUri,
        workflow=workflow,
        resources=resources,
        workDir=workDir,
        workDirPrefix="scarf-paris-",
    )


def _pseudotime_sources_sinks(
    store: DataStore, workflow: WorkflowParameters
) -> tuple[list[Any], list[Any]]:
    group_key = workflow.resolvedMarkerGroupKey
    labels = np.asarray(store.cells.fetch(group_key, key=workflow.cellKey))
    # Keep native label values (Leiden membership is int). Prefer the two
    # largest clusters so both usually sit in the retained graph component.
    values, counts = np.unique(labels, return_counts=True)
    if values.size < 2:
        raise ValueError(
            f"Need >=2 clusters in {group_key} for pseudotime; found {values.tolist()}"
        )
    order = np.argsort(-counts)
    source = values[order[0]].item()
    sink = values[order[1]].item()
    return [source], [sink]


def _peak_cgroup_bytes(measurement: ResourceMeasurement | None) -> int | None:
    if measurement is None:
        return None
    if measurement.operationPeakSource in {"cgroupMemoryCurrent", "cgroupMemoryPeak"}:
        return measurement.operationPeakBytes
    return measurement.cgroupMemoryCurrentPeakBytes


def run_stage(
    stage: StageName,
    *,
    nRows: int,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    localH5adPath: Path | None = None,
    storageLayout: StorageLayout | None = None,
    queryStoreUri: str | None = None,
    workDir: Path | None = None,
    sampleIntervalSeconds: float = 0.25,
) -> StageRunResult:
    timer = StageTimer()
    sampler = ResourceSampler(sampleIntervalSeconds=sampleIntervalSeconds)
    error: str | None = None
    status = "ok"
    measurement: ResourceMeasurement | None = None
    result_store_uri = storeUri

    with timer:
        sampler.start()
        try:
            if stage == "createStore":
                if localH5adPath is None:
                    raise ValueError("createStore requires localH5adPath")
                with timer.operation():
                    run_create_store(
                        localH5adPath=localH5adPath,
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        storageLayout=storageLayout,
                    )
            elif stage == "prepareMappingQuery":
                if localH5adPath is None:
                    raise ValueError("prepareMappingQuery requires localH5adPath")
                if queryStoreUri is None:
                    raise ValueError("prepareMappingQuery requires queryStoreUri")
                result_store_uri = queryStoreUri
                with timer.operation():
                    run_create_store(
                        localH5adPath=localH5adPath,
                        storeUri=queryStoreUri,
                        workflow=workflow,
                        resources=resources,
                        storageLayout=storageLayout,
                    )
            elif stage == "initializeStore":
                with timer.operation():
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=True
                    )
                    del store
            elif stage == "reopenStore":
                with timer.operation():
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    del store
            elif stage == "runMapping":
                if queryStoreUri is None:
                    raise ValueError("runMapping requires queryStoreUri")
                with timer.inputSetup():
                    ref = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    query = _open_datastore(
                        queryStoreUri, workflow, resources, initialize=True
                    )
                with timer.operation():
                    ref.run_mapping(
                        target_assay=_assay(query, workflow),
                        target_name=workflow.mappingTargetName,
                        target_feat_key=workflow.mappingTargetFeatKey,
                        target_cell_key=workflow.cellKey,
                        from_assay=workflow.assayName,
                        cell_key=workflow.cellKey,
                        feat_key=workflow.hvgKey,
                        save_k=workflow.mappingSaveK,
                        batch_size=workflow.mappingBatchSize,
                        missing_feature_policy="zero",
                    )
                del query
                del ref
            elif stage == "runLeiden":
                with timer.operation():
                    _run_leiden_in_subprocess(
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        workDir=workDir,
                    )
            elif stage == "runClustering":
                with timer.operation():
                    _run_paris_in_subprocess(
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        workDir=workDir,
                    )
            else:
                with timer.operation():
                    print(
                        f"[run_stage] ENTER open_datastore stage={stage} "
                        f"store={storeUri}",
                        flush=True,
                    )
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    print(
                        f"[run_stage] datastore open; ENTER analysis stage={stage}",
                        flush=True,
                    )
                    _run_analysis(
                        stage,
                        store,
                        workflow,
                        resources,
                        workDir=workDir,
                    )
                    print(f"[run_stage] analysis DONE stage={stage}", flush=True)
                    del store
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            measurement = sampler.stop()

    seconds = timer.result.measuredOperationSeconds
    peak_cgroup = _peak_cgroup_bytes(measurement)
    return StageRunResult(
        stage=stage,
        nRows=nRows,
        status=status,
        seconds=seconds,
        peakRssBytes=measurement.processTreeRssPeakBytes if measurement else None,
        peakCgroupBytes=peak_cgroup,
        modalMemoryMb=resources.modalMemoryLimitMb,
        scarfMemoryBudget=resources.scarfMemoryBudget,
        storeUri=result_store_uri,
        error=error,
        inputSetupSeconds=timer.result.inputSetupSeconds,
    )


def _run_analysis(
    stage: StageName,
    store: DataStore,
    workflow: WorkflowParameters,
    resources: StageResources,
    *,
    workDir: Path | None = None,
) -> None:
    if stage == "filterCells":
        store.auto_filter_cells(
            attrs=workflow.filterAttrs,
            min_p=workflow.filterMinQuantile,
            max_p=workflow.filterMaxQuantile,
            show_qc_plots=False,
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
        )
        return
    if stage == "makeGraph":
        store.make_graph(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            dims=workflow.dims,
            k=workflow.k,
            ann_parallel=workflow.annParallel,
            rand_state=workflow.graphSeed,
            n_centroids=workflow.nCentroids,
            show_elbow_plot=False,
            local_cache=workflow.graphLocalCache,
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
        )
        return
    if stage == "getImputed":
        for feature_name in _impute_feature_names(store, workflow):
            store.get_imputed(
                from_assay=workflow.assayName,
                cell_key=workflow.cellKey,
                feat_key=workflow.hvgKey,
                feature_name=feature_name,
                t=workflow.imputeDiffusionT,
                cache_operator=True,
            )
        return
    if stage == "runClustering":
        raise AssertionError("runClustering must execute in its child process")
    if stage == "runPseudotime":
        sources, sinks = _pseudotime_sources_sinks(store, workflow)
        store.run_pseudotime_scoring(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            source_sink_key=workflow.resolvedMarkerGroupKey,
            sources=sources,
            sinks=sinks,
            label=workflow.pseudotimeLabel,
            random_seed=workflow.leidenSeed,
        )
        return
    if stage == "makeGraphHarmony":
        _insert_synthetic_batches(store, workflow)
        store.make_graph(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            dims=workflow.dims,
            k=workflow.k,
            ann_parallel=workflow.annParallel,
            rand_state=workflow.graphSeed,
            n_centroids=workflow.nCentroids,
            show_elbow_plot=False,
            local_cache=workflow.graphLocalCache,
            harmonize=True,
            batch_columns=[workflow.harmonyBatchColumn],
        )
        return
    if stage == "subsetZarr":
        if workDir is None:
            raise ValueError("subsetZarr requires workDir")
        out = workDir / "subset.zarr"
        writer = SubsetZarr(
            zarr_loc=str(out),
            assays=[_assay(store, workflow)],
            cell_key=workflow.cellKey,
            overwrite_existing_file=True,
            overwrite_cell_data=True,
        )
        writer.dump()
        return
    if stage == "toH5ad":
        if workDir is None:
            raise ValueError("toH5ad requires workDir")
        out = workDir / "export.h5ad"
        to_h5ad(
            _assay(store, workflow),
            str(out),
            embeddings_cols=[workflow.umapLabel],
            skip_recalc_nfeats=True,
            n_threads=resources.workers,
        )
        return
    raise ValueError(f"No analysis operation for {stage}")


def repair_counts_t(
    *,
    storeUri: str,
    assayName: str,
    resources: StageResources,
    nCheckTiles: int = 3,
    seed: int = 0,
    sampleIntervalSeconds: float = 0.25,
) -> dict[str, Any]:
    """Rewrite feature-major ``countsT`` and only then mark it complete."""
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

    sampler = ResourceSampler(sampleIntervalSeconds=sampleIntervalSeconds)
    sampler.start()
    started = time.perf_counter()
    try:
        counts_t = write_counts_t(
            counts,
            group,
            profile="cloud" if storeUri.startswith("s3://") else "fast_local",
            resources=budget,
        )
    finally:
        seconds = time.perf_counter() - started
        measurement = sampler.stop()
    if counts_t is None:
        raise RuntimeError(
            f"write_counts_t skipped for {assayName} (Zarr format < 3); "
            "cannot repair countsT"
        )
    if counts_t.attrs.get("complete") is not True:
        raise RuntimeError(
            f"countsT rewrite finished without complete=True at {storeUri}"
        )

    expected_shape = (int(counts.shape[1]), int(counts.shape[0]))
    if tuple(counts_t.shape) != expected_shape:
        raise RuntimeError(
            f"countsT shape {tuple(counts_t.shape)} != expected {expected_shape}"
        )
    if np.dtype(counts_t.dtype) != np.dtype(counts.dtype):
        raise RuntimeError(
            f"countsT dtype {counts_t.dtype} != counts dtype {counts.dtype}"
        )

    rng = np.random.default_rng(seed)
    feat_chunk = max(1, int(counts_t.chunks[0]))
    cell_chunk = max(1, int(counts_t.chunks[1]))
    n_feats, n_cells = counts_t.shape
    checks: list[dict[str, Any]] = []
    for _ in range(max(0, nCheckTiles)):
        feat_start = int(rng.integers(0, n_feats))
        feat_start = (feat_start // feat_chunk) * feat_chunk
        cell_start = int(rng.integers(0, n_cells))
        cell_start = (cell_start // cell_chunk) * cell_chunk
        feat_end = min(feat_start + feat_chunk, n_feats)
        cell_end = min(cell_start + cell_chunk, n_cells)
        got = np.asarray(counts_t[feat_start:feat_end, cell_start:cell_end])
        expect = np.asarray(counts[cell_start:cell_end, feat_start:feat_end]).T
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

    return {
        "storeUri": storeUri,
        "assayName": assayName,
        "status": "ok",
        "seconds": seconds,
        "beforeComplete": before_complete,
        "complete": True,
        "shape": list(counts_t.shape),
        "chunks": list(counts_t.chunks),
        "dtype": str(counts_t.dtype),
        "workers": resources.workers,
        "peakRssBytes": measurement.processTreeRssPeakBytes if measurement else None,
        "peakCgroupBytes": _peak_cgroup_bytes(measurement),
        "checkedTiles": checks,
    }
