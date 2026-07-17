from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scarf import DataStore, H5adReader, H5adToZarr, SubsetZarr
from scarf.writers import to_h5ad

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


def _pseudotime_sources_sinks(
    store: DataStore, workflow: WorkflowParameters
) -> tuple[list[Any], list[Any]]:
    group_key = workflow.resolvedMarkerGroupKey
    labels = np.asarray(store.cells.fetch(group_key, key=workflow.cellKey))
    uniq = sorted({str(x) for x in labels.tolist()})
    if len(uniq) < 2:
        raise ValueError(
            f"Need >=2 clusters in {group_key} for pseudotime; found {uniq}"
        )
    return [uniq[0]], [uniq[-1]]


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
            else:
                with timer.operation():
                    store = _open_datastore(
                        storeUri, workflow, resources, initialize=False
                    )
                    _run_analysis(
                        stage,
                        store,
                        workflow,
                        resources,
                        workDir=workDir,
                    )
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
        store.run_clustering(
            from_assay=workflow.assayName,
            cell_key=workflow.cellKey,
            feat_key=workflow.hvgKey,
            n_clusters=workflow.parisNClusters,
            label=workflow.parisLabel,
        )
        return
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
