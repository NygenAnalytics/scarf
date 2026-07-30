"""Scanpy dask e2e stages for Modal profiling."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scarf import configure_output

from profiling.metrics import ResourceMeasurement, ResourceSampler, StageTimer
from profiling.scanpy_config import (
    SCANPY_STAGE_ORDER,
    ScanpyDaskParameters,
    ScanpyModalResources,
    ScanpyProfilingConfig,
    ScanpyStageName,
    ScanpyWorkflowParameters,
)

configure_output(progress=False, timestamps=True)


def _peak_cgroup_bytes(measurement: ResourceMeasurement | None) -> int | None:
    if measurement is None:
        return None
    if measurement.operationPeakBytes is not None:
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
class ScanpyStageRunResult:
    stage: ScanpyStageName
    nRows: int
    status: str
    seconds: float | None
    peakRssBytes: int | None
    peakCgroupBytes: int | None
    modalMemoryMb: int
    error: str | None = None
    inputSetupSeconds: float | None = None
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


@dataclass
class ScanpyPipelineState:
    """Mutable state shared across sequential Scanpy stages."""

    adata: Any | None = None
    h5File: Any | None = None
    client: Any | None = None
    cluster: Any | None = None
    nObsAfterFilter: int | None = None
    nVarHvgs: int | None = None
    nClusters: int | None = None

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self.client = None
        if self.cluster is not None:
            try:
                self.cluster.close()
            except Exception:  # noqa: BLE001
                pass
            self.cluster = None
        if self.h5File is not None:
            try:
                self.h5File.close()
            except Exception:  # noqa: BLE001
                pass
            self.h5File = None


def _feature_names(adata: Any, workflow: ScanpyWorkflowParameters) -> pd.Index:
    if workflow.featureNameKey in adata.var.columns:
        return pd.Index(adata.var[workflow.featureNameKey].astype(str))
    return adata.var_names.astype(str)


def _mark_qc_gene_sets(
    adata: Any,
    workflow: ScanpyWorkflowParameters,
) -> None:
    names = _feature_names(adata, workflow)
    adata.var["mt"] = names.str.contains(workflow.mitoPattern, regex=True, na=False)
    adata.var["ribo"] = names.str.contains(workflow.riboPattern, regex=True, na=False)


def _start_dask(dask: ScanpyDaskParameters) -> tuple[Any, Any]:
    from dask.distributed import Client, LocalCluster

    cluster_kwargs: dict[str, Any] = {
        "n_workers": dask.nWorkers,
        "threads_per_worker": dask.threadsPerWorker,
        "processes": dask.processes,
        "memory_limit": dask.memoryPerWorker,
        "death_timeout": dask.deathTimeoutSeconds,
    }
    if dask.dashboardAddress is not None:
        cluster_kwargs["dashboard_address"] = dask.dashboardAddress
    cluster = LocalCluster(**cluster_kwargs)
    client = Client(cluster)
    return cluster, client


def _load_lazy_h5ad(
    path: Path,
    *,
    chunkSize: int,
) -> tuple[Any, Any]:
    """Open H5AD counts lazily as a sparse-chunked dask array.

    The returned file handle must stay open for the lifetime of ``adata.X``.
    Prepared Scarf profiling subsets store raw counts under ``/X``.
    """
    import anndata as ad
    import h5py

    h5_file = h5py.File(path, "r")
    try:
        obs = ad.io.read_elem(h5_file["obs"])
        var = ad.io.read_elem(h5_file["var"])
        matrix_key = "raw/X" if "raw" in h5_file and "X" in h5_file["raw"] else "X"
        n_vars = int(var.shape[0])
        x = ad.experimental.read_elem_lazy(
            h5_file[matrix_key],
            chunks=(chunkSize, n_vars),
        )
        adata = ad.AnnData(obs=obs, var=var)
        adata.X = x
        return adata, h5_file
    except Exception:
        h5_file.close()
        raise


def _quantile_cell_mask(
    obs: pd.DataFrame,
    columns: list[str],
    *,
    minQuantile: float,
    maxQuantile: float,
) -> pd.Series:
    mask = pd.Series(True, index=obs.index)
    for column in columns:
        values = obs[column].to_numpy()
        lo, hi = np.quantile(values, [minQuantile, maxQuantile])
        mask &= (obs[column] >= lo) & (obs[column] <= hi)
    return mask


def _run_stage_body(
    stage: ScanpyStageName,
    state: ScanpyPipelineState,
    workflow: ScanpyWorkflowParameters,
    dask: ScanpyDaskParameters,
    localH5adPath: Path | None,
) -> dict[str, Any]:
    import scanpy as sc

    details: dict[str, Any] = {}
    if stage == "loadLazy":
        if localH5adPath is None:
            raise ValueError("loadLazy requires localH5adPath")
        cluster, client = _start_dask(dask)
        state.cluster = cluster
        state.client = client
        adata, h5_file = _load_lazy_h5ad(
            localH5adPath,
            chunkSize=dask.sparseChunkSize,
        )
        state.adata = adata
        state.h5File = h5_file
        details["shape"] = list(adata.shape)
        details["xType"] = type(adata.X).__name__
        details["daskWorkers"] = dask.nWorkers
        details["sparseChunkSize"] = dask.sparseChunkSize
        details["scheduler"] = client.scheduler.address
        return details

    if state.adata is None:
        raise RuntimeError(f"{stage} requires a loaded AnnData")
    adata = state.adata

    if stage == "calculateQc":
        _mark_qc_gene_sets(adata, workflow)
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "ribo"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        details["nMtGenes"] = int(adata.var["mt"].sum())
        details["nRiboGenes"] = int(adata.var["ribo"].sum())
        return details

    if stage == "filterCells":
        before = int(adata.n_obs)
        sc.pp.filter_cells(adata, min_genes=workflow.minGenesPerCell)
        sc.pp.filter_genes(adata, min_cells=workflow.minCellsPerGene)
        qc_cols = [
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "pct_counts_ribo",
        ]
        missing = [col for col in qc_cols if col not in adata.obs.columns]
        if missing:
            raise RuntimeError(
                "calculateQc must run before filterCells; missing "
                + ", ".join(missing)
            )
        keep = _quantile_cell_mask(
            adata.obs,
            qc_cols,
            minQuantile=workflow.filterMinQuantile,
            maxQuantile=workflow.filterMaxQuantile,
        )
        # Keep the subset lazy: AnnData.copy() would materialize dask X.
        filtered = adata[keep.to_numpy(), :]
        state.adata = filtered
        state.nObsAfterFilter = int(filtered.n_obs)
        details["nObsBefore"] = before
        details["nObsAfter"] = int(filtered.n_obs)
        details["nVar"] = int(filtered.n_vars)
        return details

    if stage == "normalizeTotal":
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=workflow.targetSum)
        details["targetSum"] = workflow.targetSum
        return details

    if stage == "log1p":
        sc.pp.log1p(adata)
        return details

    if stage == "markHvgs":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=workflow.nTopGenes,
            flavor=workflow.hvgFlavor,
        )
        state.nVarHvgs = int(adata.var["highly_variable"].sum())
        details["nTopGenes"] = workflow.nTopGenes
        details["nHvgs"] = state.nVarHvgs
        details["flavor"] = workflow.hvgFlavor
        return details

    if stage == "runPca":
        sc.pp.pca(
            adata,
            n_comps=workflow.nComps,
            mask_var="highly_variable",
        )
        x_pca = adata.obsm["X_pca"]
        if hasattr(x_pca, "compute"):
            adata.obsm["X_pca"] = x_pca.compute()
        details["nComps"] = workflow.nComps
        details["pcaShape"] = list(np.asarray(adata.obsm["X_pca"]).shape)
        return details

    if stage == "runNeighbors":
        transformer = None
        if workflow.neighborsTransformer == "annoy":
            from sklearn_ann.kneighbors.annoy import AnnoyTransformer

            transformer = AnnoyTransformer(n_neighbors=workflow.nNeighbors)
        sc.pp.neighbors(
            adata,
            n_neighbors=workflow.nNeighbors,
            n_pcs=workflow.nComps,
            transformer=transformer,
        )
        details["nNeighbors"] = workflow.nNeighbors
        details["transformer"] = workflow.neighborsTransformer
        return details

    if stage == "runUmap":
        sc.tl.umap(
            adata,
            maxiter=workflow.umapEpochs,
            random_state=workflow.umapSeed,
        )
        details["umapEpochs"] = workflow.umapEpochs
        return details

    if stage == "runLeiden":
        sc.tl.leiden(
            adata,
            resolution=workflow.leidenResolution,
            random_state=workflow.leidenSeed,
            flavor=workflow.leidenFlavor,
            n_iterations=workflow.leidenNIterations,
            key_added=workflow.leidenKeyAdded,
        )
        labels = adata.obs[workflow.leidenKeyAdded]
        state.nClusters = int(labels.nunique())
        details["resolution"] = workflow.leidenResolution
        details["flavor"] = workflow.leidenFlavor
        details["nClusters"] = state.nClusters
        return details

    if stage == "rankGenesGroups":
        rank_kwargs: dict[str, Any] = {
            "groupby": workflow.leidenKeyAdded,
            "method": workflow.rankMethod,
            "use_raw": False,
        }
        if workflow.rankNGenes is not None:
            rank_kwargs["n_genes"] = workflow.rankNGenes
        sc.tl.rank_genes_groups(adata, **rank_kwargs)
        details["method"] = workflow.rankMethod
        details["groupby"] = workflow.leidenKeyAdded
        return details

    raise ValueError(f"Unknown Scanpy stage: {stage}")


def run_scanpy_stage(
    stage: ScanpyStageName,
    *,
    nRows: int,
    state: ScanpyPipelineState,
    workflow: ScanpyWorkflowParameters,
    dask: ScanpyDaskParameters,
    resources: ScanpyModalResources,
    localH5adPath: Path | None = None,
    resetCgroupPeak: bool = True,
) -> ScanpyStageRunResult:
    sampler = ResourceSampler(resetCgroupPeak=resetCgroupPeak)
    sampler.start()
    timer = StageTimer()
    error: str | None = None
    details: dict[str, Any] | None = None
    status = "ok"
    try:
        with timer:
            with timer.operation():
                details = _run_stage_body(
                    stage,
                    state,
                    workflow,
                    dask,
                    localH5adPath,
                )
    except Exception as exc:  # noqa: BLE001 - persist stage failure
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        measurement = sampler.stop()

    timings = timer.result
    summary = summarize_resource_measurement(measurement)
    return ScanpyStageRunResult(
        stage=stage,
        nRows=nRows,
        status=status,
        seconds=timings.measuredOperationSeconds,
        peakRssBytes=summary["peakRssBytes"],
        peakCgroupBytes=summary["peakCgroupBytes"],
        modalMemoryMb=resources.modalMemoryLimitMb,
        error=error,
        inputSetupSeconds=timings.inputSetupSeconds,
        wholeFunctionSeconds=timings.wholeFunctionSeconds,
        modalCpuRequest=resources.modalCpuRequest,
        modalCpuLimit=resources.modalCpuLimit,
        rssBaselineBytes=summary["rssBaselineBytes"],
        rssIncrementalPeakBytes=summary["rssIncrementalPeakBytes"],
        rssAfterBytes=summary["rssAfterBytes"],
        cgroupCurrentBaselineBytes=summary["cgroupCurrentBaselineBytes"],
        cgroupCurrentPeakBytes=summary["cgroupCurrentPeakBytes"],
        cgroupCurrentAfterBytes=summary["cgroupCurrentAfterBytes"],
        operationBaselineBytes=summary["operationBaselineBytes"],
        operationIncrementalPeakBytes=summary["operationIncrementalPeakBytes"],
        operationPeakSource=summary["operationPeakSource"],
        cgroupPeakScope=summary["cgroupPeakScope"],
        details=details,
    )


def write_scanpy_result(config: ScanpyProfilingConfig, result: ScanpyStageRunResult) -> str:
    from profiling.r2 import put_json

    uri = config.resultUri(result.nRows, result.stage)
    put_json(uri, result.to_json())
    return uri


def write_scanpy_funnel_result(
    config: ScanpyProfilingConfig,
    nRows: int,
    payload: dict[str, object],
) -> str:
    from profiling.r2 import put_json

    uri = config.funnelResultUri(nRows)
    put_json(uri, payload)
    return uri


def comparison_notes() -> dict[str, Any]:
    """Document fairness choices vs Scarf CORE e2e."""
    return {
        "scarfConversionIncluded": [
            "createStore",
            "writeCountsT",
        ],
        "scanpyConversion": "none (H5AD lazy open only)",
        "normalizeTargetSum": 1e4,
        "scarfNormalizeSf": 1000,
        "hvgCount": 2000,
        "pcaComps": 50,
        "neighborsK": 11,
        "leidenResolution": 1.0,
        "umapEpochs": 300,
        "paris": "skipped on both sides for Scanpy peer",
        "cellFilter": "approximate 1st/99th quantiles on Scanpy QC metrics",
        "daskScope": "lazy through HVG+PCA; neighbors/UMAP/Leiden on in-memory X_pca",
    }


def run_scanpy_e2e_funnel_body(
    config: ScanpyProfilingConfig,
    nRows: int,
    *,
    workDir: Path,
) -> dict[str, Any]:
    from profiling.r2 import download_file, put_json_if_absent

    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    if not config.runTag.strip():
        raise ValueError("run-e2e requires a non-empty runTag")

    claimed = put_json_if_absent(
        config.e2eClaimUri(),
        {
            "runTag": config.runTag,
            "nRows": nRows,
            "status": "claimed",
            "stack": "scanpy",
        },
    )
    if not claimed:
        raise FileExistsError(
            f"run-e2e runTag was claimed concurrently: {config.runTag}"
        )

    workDir.mkdir(parents=True, exist_ok=True)
    local_h5ad = workDir / f"{nRows}.h5ad"
    state = ScanpyPipelineState()
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    download_seconds: float | None = None
    outcomes: list[dict[str, Any]] = []
    completed_stages: list[ScanpyStageName] = []
    status = "ok"
    error: str | None = None
    failed_stage: ScanpyStageName | None = None
    try:
        download_started = time.perf_counter()
        print(f"scanpy e2e dataset download start: {config.datasetUri(nRows)}", flush=True)
        download_file(config.datasetUri(nRows), local_h5ad)
        download_seconds = time.perf_counter() - download_started
        print(
            f"scanpy e2e dataset download done: seconds={download_seconds:.1f}",
            flush=True,
        )

        for stage in SCANPY_STAGE_ORDER:
            failed_stage = stage
            print(f"scanpy e2e stage start: {stage}", flush=True)
            result = run_scanpy_stage(
                stage,
                nRows=nRows,
                state=state,
                workflow=config.workflow,
                dask=config.dask,
                resources=config.resources,
                localH5adPath=local_h5ad if stage == "loadLazy" else None,
                resetCgroupPeak=False,
            )
            result_uri = write_scanpy_result(config, result)
            payload = result.to_json()
            payload["resultUri"] = result_uri
            outcomes.append(payload)
            print(
                f"scanpy e2e stage done: {stage} status={result.status} "
                f"seconds={result.seconds}",
                flush=True,
            )
            if result.status != "ok":
                status = "error"
                error = result.error or f"{stage} failed"
                break
            completed_stages.append(stage)
        else:
            failed_stage = None
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        measurement = sampler.stop()
        state.close()

    summary: dict[str, Any] = {
        "runTag": config.runTag,
        "stack": "scanpy",
        "nRows": nRows,
        "status": status,
        "stopped": status != "ok",
        "error": error,
        "failedStage": failed_stage,
        "datasetUri": config.datasetUri(nRows),
        "datasetDownloadSeconds": download_seconds,
        "wholeFunctionSeconds": time.perf_counter() - started,
        "modalResources": config.resources.model_dump(mode="python"),
        "dask": config.dask.model_dump(mode="python"),
        "workflow": config.workflow.model_dump(mode="python"),
        "stageOrder": list(SCANPY_STAGE_ORDER),
        "completedStages": completed_stages,
        "outcomes": outcomes,
        "claimUri": config.e2eClaimUri(),
        "funnelResultUri": config.funnelResultUri(nRows),
        "nObsAfterFilter": state.nObsAfterFilter,
        "nVarHvgs": state.nVarHvgs,
        "nClusters": state.nClusters,
        "comparisonNotes": comparison_notes(),
        **summarize_resource_measurement(measurement),
    }
    write_scanpy_funnel_result(config, nRows, summary)
    return summary
