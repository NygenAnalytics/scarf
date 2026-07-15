from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scarf import DataStore, H5adReader, H5adToZarr

from profiling.config import (
    StageName,
    StageResources,
    StorageLayout,
    WorkflowParameters,
)
from profiling.metrics import ResourceMeasurement, ResourceSampler, StageTimer
from profiling.r2 import storage_options


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
        "zarrProfile": "cloud" if storeUri.startswith("s3://") else None,
        "storage_options": options,
        "mem_budget": resources.scarfMemoryBudget,
        "working_copies": resources.workingCopies,
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
    if storageLayout is not None and storageLayout.targetChunkBytes is not None:
        layout_kwargs = {
            "targetChunkBytes": storageLayout.targetChunkBytes,
            "minFeatureChunk": storageLayout.minFeatureChunk,
            "maxFeatureChunk": storageLayout.maxFeatureChunk,
        }
    try:
        writer = H5adToZarr(
            reader,
            storeUri,
            assay_name=workflow.assayName,
            storage_options=options,
            mem_budget=resources.scarfMemoryBudget,
            nthreads=resources.workers,
            working_copies=resources.workingCopies,
            **layout_kwargs,
        )
        writer.dump(batch_size=workflow.h5adBatchSize)
    finally:
        if hasattr(reader, "h5") and hasattr(reader.h5, "close"):
            reader.h5.close()


def run_stage(
    stage: StageName,
    *,
    nRows: int,
    storeUri: str,
    workflow: WorkflowParameters,
    resources: StageResources,
    localH5adPath: Path | None = None,
    storageLayout: StorageLayout | None = None,
    sampleIntervalSeconds: float = 0.25,
) -> StageRunResult:
    timer = StageTimer()
    sampler = ResourceSampler(sampleIntervalSeconds=sampleIntervalSeconds)
    error: str | None = None
    status = "ok"
    measurement: ResourceMeasurement | None = None

    with timer:
        sampler.start()
        try:
            with timer.operation():
                if stage == "createStore":
                    if localH5adPath is None:
                        raise ValueError("createStore requires localH5adPath")
                    run_create_store(
                        localH5adPath=localH5adPath,
                        storeUri=storeUri,
                        workflow=workflow,
                        resources=resources,
                        storageLayout=storageLayout,
                    )
                elif stage == "initializeStore":
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=True
                    )
                    del store
                elif stage == "reopenStore":
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    del store
                else:
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    _run_analysis(stage, store, workflow, resources)
                    del store
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            measurement = sampler.stop()

    seconds = timer.result.measuredOperationSeconds
    peak_cgroup = None
    if measurement is not None:
        if measurement.operationPeakSource in {
            "cgroupMemoryCurrent",
            "cgroupMemoryPeak",
        }:
            peak_cgroup = measurement.operationPeakBytes
        else:
            peak_cgroup = measurement.cgroupMemoryCurrentPeakBytes
    return StageRunResult(
        stage=stage,
        nRows=nRows,
        status=status,
        seconds=seconds,
        peakRssBytes=measurement.processTreeRssPeakBytes if measurement else None,
        peakCgroupBytes=peak_cgroup,
        modalMemoryMb=resources.modalMemoryLimitMb,
        scarfMemoryBudget=resources.scarfMemoryBudget,
        storeUri=storeUri,
        error=error,
    )


def _run_analysis(
    stage: StageName,
    store: DataStore,
    workflow: WorkflowParameters,
    resources: StageResources,
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
        store.run_leiden_clustering(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            resolution=workflow.leidenResolution,
            label=workflow.leidenLabel,
            random_seed=workflow.leidenSeed,
        )
        return
    if stage == "findMarkers":
        store.run_marker_search(
            from_assay=workflow.assayName,
            group_key=workflow.resolvedMarkerGroupKey,
            cell_key=workflow.cellKey,
            feat_key=workflow.markerFeatureKey,
            gene_batch_size=workflow.markerGeneBatchSize,
            use_prenormed=False,
            prenormed_store=None,
            n_threads=resources.workers,
            skip_save=False,
        )
        return
    raise ValueError(f"No analysis operation for {stage}")
