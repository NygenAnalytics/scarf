from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..embeddings.harmony import HarmonyResult
from ..matrix import ChunkedArray
from ..utils.logging import logger
from .stages import (
    AnnIndexStage,
    BatchCorrectionStage,
    ChunkedCoordinateStream,
    CoordinateSource,
    KMeansInitializationStage,
    LazyTransformStream,
    NeighborQueryStage,
    ReductionTransform,
)

__all__ = ["AnnStream"]


class AnnStream:
    """Compatibility adapter over lazy graph-computation stages."""

    reducer: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self,
        data: ChunkedArray,
        k: int,
        n_cluster: int,
        reduction_method: str,
        dims: int | None,
        loadings: np.ndarray | None,
        use_for_pca: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        ann_metric: str,
        ann_efc: int,
        ann_ef: int,
        ann_m: int,
        nthreads: int,
        ann_parallel: bool,
        rand_state: int,
        do_kmeans_fit: bool,
        disable_scaling: bool,
        ann_idx: Any | None,
        lsi_skip_first: bool,
        lsi_params: dict[str, Any],
        harmonize: bool,
        harmonized_data: ChunkedArray | None = None,
        batches: pd.DataFrame | None = None,
        harmony_params: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.k = min(k, self.data.shape[0] - 1)
        self.nClusters = max(n_cluster, 2)
        self.dims = dims
        self.loadings = loadings
        if self.dims is None and self.loadings is None:
            raise ValueError(
                "ERROR: Provide either value for atleast one: 'dims' or 'loadings'"
            )
        self.annMetric = ann_metric
        self.annEfc = ann_efc
        self.annEf = ann_ef
        self.annM = ann_m
        self.nthreads = nthreads
        self.annThreads = self.nthreads if ann_parallel else 1
        self.randState = rand_state
        self.batchSize = self._handle_batch_size()
        self.method = reduction_method
        self.nCells, self.nFeats = self.data.shape
        self.clusterLabels: np.ndarray = np.repeat(-1, self.nCells)
        self.harmonize = harmonize
        self.harmonizedData = harmonized_data
        self.harmonyResult: HarmonyResult | None = None
        self.batches = batches
        self.harmonyParams = dict(harmony_params or {})
        self.featureScaling = not disable_scaling

        with threadpool_limits(limits=self.nthreads):
            self.reduction = ReductionTransform(
                data=self.data,
                method=self.method,
                dims=self.dims,
                loadings=self.loadings,
                use_for_pca=use_for_pca,
                mu=mu,
                sigma=sigma,
                batch_size=self.batchSize,
                nthreads=self.nthreads,
                rand_state=self.randState,
                disable_scaling=disable_scaling,
                lsi_skip_first=lsi_skip_first,
                lsi_params=lsi_params,
            )
            self.dims = self.reduction.dims
            self.loadings = self.reduction.loadings
            self.reducer = self.reduction.transform
            if self.method == "pca":
                self.mu = self.reduction.mu
                self.sigma = self.reduction.sigma
                if self.reduction.pca is not None:
                    self._pca = self.reduction.pca

            self.transform_stream = LazyTransformStream(
                data=self.data,
                transform=self.reducer,
                nthreads=self.nthreads,
                batch_size=self.batchSize,
            )
            self.batch_correction = (
                BatchCorrectionStage(
                    stream=self.transform_stream,
                    n_cells=self.nCells,
                    dims=int(self.dims or self.nFeats),
                    batch_size=self.batchSize,
                    batches=self.batches,
                    parameters=self.harmonyParams,
                    corrected_data=self.harmonizedData,
                    nthreads=self.nthreads,
                )
                if self.harmonize
                else None
            )
            if ann_idx is None:
                self.annIdx = self._fit_ann()
            else:
                self.annIdx = AnnIndexStage.configure(
                    ann_idx,
                    ef=self.annEf,
                    threads=self.annThreads,
                )
            self.neighbor_query = NeighborQueryStage(
                self.annIdx,
                self.k,
                self.annMetric,
            )
            self._sync_batch_correction()
            self.kmeans = self._fit_kmeans(do_kmeans_fit)

    def _sync_batch_correction(self) -> None:
        if self.batch_correction is None:
            return
        self.harmonizedData = self.batch_correction.corrected_data
        self.harmonyResult = self.batch_correction.result

    def iter_blocks(self, msg: str = "") -> Generator[np.ndarray, None, None]:
        yield from self.transform_stream.iter_raw(msg)

    def transform_query(self, values: np.ndarray) -> np.ndarray:
        query = np.asarray(values)
        if query.ndim != 2:
            raise ValueError("Query data must be a two-dimensional array")
        if query.shape[1] != self.nFeats:
            raise ValueError(
                f"Query has {query.shape[1]} features but reference expects {self.nFeats}"
            )
        result = np.asarray(self.reducer(query))
        expected_dims = (
            self.dims if self.dims is not None and self.dims > 0 else self.nFeats
        )
        if result.ndim != 2 or result.shape[1] != expected_dims:
            raise ValueError("Query transform did not produce the ANN index dimensions")
        if not np.all(np.isfinite(result)):
            raise ValueError("Query transform produced non-finite values")
        return result

    def transform_ann(
        self,
        values: np.ndarray,
        k: int | None = None,
        self_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        return self.neighbor_query.query(
            values,
            k=k,
            self_indices=self_indices,
        )

    def _fit_ann(self) -> Any:
        ann_dims = (
            self.dims
            if self.dims is not None and self.dims >= 1
            else self.data.shape[1]
        )
        index = AnnIndexStage.create(
            metric=self.annMetric,
            dims=ann_dims,
            n_cells=self.nCells,
            ef_construction=self.annEfc,
            ef=self.annEf,
            m=self.annM,
            rand_state=self.randState,
            nthreads=self.annThreads,
        )
        coordinates: CoordinateSource = self.transform_stream
        if self.batch_correction is not None:
            coordinates = ChunkedCoordinateStream(
                self.batch_correction.ensure_corrected(),
                self.nthreads,
            )
        index = AnnIndexStage.populate(index, coordinates)
        self._sync_batch_correction()
        return index

    def _fit_kmeans(self, enabled: bool) -> Any | None:
        result = KMeansInitializationStage.fit(
            stream=self.transform_stream,
            n_rows=self.nCells,
            batch_size=self.batchSize,
            n_clusters=self.nClusters,
            rand_state=self.randState,
            nthreads=self.nthreads,
            enabled=enabled,
        )
        self.clusterLabels = result.labels
        return result.model

    def _handle_batch_size(self) -> int:
        if self.dims is not None and self.dims > self.data.shape[0]:
            self.dims = self.data.shape[0]
        batch_size = self.data.chunksize[0]
        if self.dims is not None and self.dims >= batch_size:
            self.dims = batch_size - 1
            logger.warning(
                f"Number of PCA/LSI components reduced to batch size of {batch_size}"
            )
        if self.nClusters > batch_size:
            self.nClusters = batch_size
            logger.warning(f"Cluster number reduced to batch size of {batch_size}")
        return batch_size
