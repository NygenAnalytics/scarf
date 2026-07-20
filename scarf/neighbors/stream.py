from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..matrix import ChunkedArray
from ..embeddings.harmony import HarmonyResult, fit_harmony
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.progress import tqdmbar
from .index import (
    fix_knn_query as _fix_knn_query,
    instantiate_knn_index as _instantiate_knn_index,
)

__all__ = ["AnnStream"]


class AnnStream:
    """Stream row blocks through dimensionality reduction and fit ANN / k-means.

    Args:
        data: ChunkedArray of cell-by-feature counts.
        k: Number of nearest neighbors to query.
        n_cluster: Number of k-means clusters for seed partitions.
        reduction_method: One of ``'pca'``, ``'lsi'``, or ``'custom'``.
        dims: Number of reduced dimensions (or loadings columns) to use.
        loadings: Precomputed loading matrix; if None, loadings are fit from data.
        use_for_pca: Boolean mask of cells used when fitting PCA.
        mu: Feature means for scaling (PCA).
        sigma: Feature std devs for scaling (PCA).
        ann_metric: hnswlib distance metric.
        ann_efc: hnswlib ``ef_construction``.
        ann_ef: hnswlib ``ef`` search parameter.
        ann_m: hnswlib ``M`` connectivity parameter.
        nthreads: Thread limit for sklearn / block compute.
        ann_parallel: If True, use ``nthreads`` for hnswlib as well.
        rand_state: Random seed.
        do_kmeans_fit: Whether to fit MiniBatchKMeans seed partitions.
        disable_scaling: Skip z-scaling before PCA projection.
        ann_idx: Existing hnswlib index to reuse instead of fitting.
        lsi_skip_first: Drop first LSI component (depth) when using LSI.
        lsi_params: Extra kwargs forwarded to sklearn TruncatedSVD (minus reserved keys).
        harmonize: Whether to run Harmony batch correction before ANN fitting.
        harmonized_data: Precomputed harmonized ChunkedArray (optional).
        batches: Batch metadata DataFrame for Harmony.
    """

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
        cache_embeddings: bool = True,
        harmony_params: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self._embeddings: np.ndarray | None = None
        self.k = k
        if self.k >= self.data.shape[0]:
            self.k = self.data.shape[0] - 1
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
        if ann_parallel:
            self.annThreads = self.nthreads
        else:
            self.annThreads = 1
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
        disable_reduction = False
        if self.dims is not None and self.dims < 1:
            disable_reduction = True
        with threadpool_limits(limits=self.nthreads):
            if self.method == "pca":
                self.mu, self.sigma = mu, sigma
                if self.loadings is None or len(self.loadings) == 0:
                    if len(use_for_pca) != self.nCells:
                        raise ValueError(
                            "ERROR: `use_for_pca` does not have sample length as nCells"
                        )
                    if disable_reduction is False:
                        self._fit_pca(disable_scaling, use_for_pca)
                else:
                    # Even though the dims might have been already adjusted according to loadings before calling
                    # AnnStream, it could still be overwritten by _handle_batch_size. Hence need to hard set it here.
                    self.dims = self.loadings.shape[1]
                    # it is okay for dimensions to be larger than batch size here because we will not fit the PCA
                loadings = self.loadings
                if disable_scaling:
                    if disable_reduction:

                        def reducer(x: np.ndarray) -> np.ndarray:
                            return x
                    else:
                        assert loadings is not None
                        loadings_mat = loadings

                        def reducer(x: np.ndarray) -> np.ndarray:
                            return np.asarray(x.dot(loadings_mat))
                else:
                    if disable_reduction:

                        def reducer(x: np.ndarray) -> np.ndarray:
                            return self.transform_z(x)
                    else:
                        assert loadings is not None
                        loadings_mat = loadings

                        def reducer(x: np.ndarray) -> np.ndarray:
                            return np.asarray(self.transform_z(x).dot(loadings_mat))

                self.reducer = reducer
            elif self.method == "lsi":
                if self.loadings is None or len(self.loadings) == 0:
                    if disable_reduction is False:
                        self._fit_lsi(lsi_skip_first, lsi_params)
                else:
                    # First dimension of LSI captures depth
                    if lsi_skip_first:
                        self.loadings = self.loadings[:, 1:]
                    self.dims = self.loadings.shape[1]
                loadings = self.loadings
                if disable_reduction:

                    def reducer(x: np.ndarray) -> np.ndarray:
                        return x
                else:
                    assert loadings is not None
                    loadings_mat = loadings

                    def reducer(x: np.ndarray) -> np.ndarray:
                        return np.asarray(x.dot(loadings_mat))

                self.reducer = reducer
            elif self.method == "custom":
                if self.loadings is None or len(self.loadings) == 0:
                    logger.warning(
                        "No loadings provided for manual dimension reduction"
                    )
                else:
                    self.dims = self.loadings.shape[1]
                loadings = self.loadings
                if disable_reduction:

                    def reducer(x: np.ndarray) -> np.ndarray:
                        return x
                else:
                    assert loadings is not None
                    loadings_mat = loadings

                    def reducer(x: np.ndarray) -> np.ndarray:
                        return np.asarray(x.dot(loadings_mat))

                self.reducer = reducer
            else:
                raise ValueError(f"ERROR: Unknown reduction method: {self.method}")
            if cache_embeddings:
                self._maybe_build_embeddings()
            if ann_idx is None:
                self.annIdx = self._fit_ann()
            else:
                self.annIdx = ann_idx
                self.annIdx.set_ef(self.annEf)
                self.annIdx.set_num_threads(self.annThreads)
            self.kmeans = self._fit_kmeans(do_kmeans_fit)

    @property
    def embeddings(self) -> np.ndarray | None:
        """Reduced cell embeddings when cached in memory, else None."""
        return self._embeddings

    def _reduced_blocks(self, msg: str) -> list[np.ndarray]:
        """Apply the reducer to every row block in parallel, preserving order."""
        return self.data.map_blocks(
            lambda _i, s, e: self.reducer(self.data._materialize_range(s, e)),
            nthreads=self.nthreads,
            msg=msg,
        )

    def _maybe_build_embeddings(self) -> None:
        if self.harmonize:
            return
        self._embeddings = np.vstack(
            self._reduced_blocks(msg="Building cell embeddings")
        )

    def _handle_batch_size(self) -> int:
        if self.dims is not None and self.dims > self.data.shape[0]:
            self.dims = self.data.shape[0]
        batch_size = self.data.chunksize[0]  # Assuming all chunks are same size
        if self.dims is not None and self.dims >= batch_size:
            self.dims = batch_size - 1  # -1 because we will do PCA +1
            logger.info(
                f"Number of PCA/LSI components reduced to batch size of {batch_size}"
            )
        if self.nClusters > batch_size:
            self.nClusters = batch_size
            logger.info(f"Cluster number reduced to batch size of {batch_size}")
        return batch_size

    def iter_blocks(self, msg: str = "") -> Generator[np.ndarray, None, None]:
        """Yield row blocks of raw data as NumPy arrays with optional progress bar."""
        yield from self.data.stream_blocks(nthreads=self.nthreads, msg=msg)

    def transform_z(self, a: np.ndarray) -> np.ndarray:
        """Z-score a block using fitted ``mu`` and ``sigma``."""
        return np.asarray((a - self.mu) / self.sigma)

    def transform_query(self, a: np.ndarray) -> np.ndarray:
        """Project query feature rows into this index's native latent space."""
        values = np.asarray(a)
        if values.ndim != 2:
            raise ValueError("Query data must be a two-dimensional array")
        if values.shape[1] != self.nFeats:
            raise ValueError(
                f"Query has {values.shape[1]} features but reference expects {self.nFeats}"
            )
        result = np.asarray(self.reducer(values))
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
        a: np.ndarray,
        k: int | None = None,
        self_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        """Query the ANN index for neighbors of transformed rows in ``a``.

        Args:
            a: Transformed embedding block, shape (n_rows, n_dims).
            k: Number of neighbors; defaults to ``self.k``.
            self_indices: Global row indices for self-neighbor correction.

        Returns:
            Tuple of (indices, distances) from hnswlib, optionally corrected
            when ``self_indices`` is provided.
        """
        if k is None:
            k = self.k
        # Adding +1 to k because first neighbour will be the query itself
        if self_indices is None:
            i, d = self.annIdx.knn_query(a, k=k)
            return np.asarray(i), np.asarray(d)
        i, d = self.annIdx.knn_query(a, k=k + 1)
        return _fix_knn_query(i, d, self_indices)

    def _fit_pca(self, disable_scaling: bool, use_for_pca: np.ndarray) -> None:
        from ..embeddings.reduction import fit_incremental_pca

        assert self.dims is not None
        scale = self.transform_z if disable_scaling is False else None
        self.loadings, self._pca = fit_incremental_pca(
            self.data,
            dims=self.dims,
            batch_size=self.batchSize,
            use_for_pca=use_for_pca,
            scale=scale,
            nthreads=self.nthreads,
        )

    def _fit_lsi(self, lsi_skip_first: bool, lsi_params: dict[str, Any]) -> None:
        from ..embeddings.reduction import fit_lsi

        assert self.dims is not None
        self.loadings = fit_lsi(
            self.data,
            dims=self.dims,
            skip_first=lsi_skip_first,
            params=lsi_params,
            random_state=self.randState,
            nthreads=self.nthreads,
        )

    def _fit_ann(self) -> Any:
        def _transform_values() -> np.ndarray:
            return np.vstack(
                self._reduced_blocks(msg="Calculating uncorrected latent dimensions")
            ).T

        ann_dims = (
            self.dims
            if self.dims is not None and self.dims >= 1
            else self.data.shape[1]
        )

        ann_idx = _instantiate_knn_index(
            self.annMetric,
            ann_dims,
            self.nCells,
            self.annEfc,
            self.annM,
            self.randState,
            self.annEf,
            self.annThreads,
        )
        if self.harmonize:
            if self.harmonizedData is None:
                if self.batches is None:
                    raise ValueError("Harmony requires batch metadata")
                self.harmonyResult = fit_harmony(
                    _transform_values(), self.batches, **self.harmonyParams
                )
                self.harmonizedData = ChunkedArray.from_numpy(
                    self.harmonyResult.corrected.T,
                    block_size=self.data.chunksize[0],
                    nthreads=self.nthreads,
                )
            for i in tqdmbar(
                self.harmonizedData.blocks,
                desc="Fitting ANN",
                total=self.harmonizedData.numblocks[0],
            ):
                ann_idx.add_items(controlled_compute(i, self.nthreads))
        elif self._embeddings is not None:
            bs = self.batchSize
            n_blocks = int(np.ceil(self.nCells / bs))
            for start in tqdmbar(
                range(0, self.nCells, bs),
                desc="Fitting ANN",
                total=n_blocks,
            ):
                end = min(start + bs, self.nCells)
                ann_idx.add_items(self._embeddings[start:end])
        else:
            for i in self.iter_blocks(msg="Fitting ANN"):
                ann_idx.add_items(self.reducer(i))
        return ann_idx

    def _fit_kmeans(self, do_ann_fit: bool) -> Any | None:
        from sklearn.cluster import MiniBatchKMeans

        if do_ann_fit is False:
            return None
        kmeans = MiniBatchKMeans(
            n_clusters=self.nClusters,
            random_state=self.randState,
            batch_size=self.batchSize,
            n_init=3,
        )
        temp: list[int] = []
        with threadpool_limits(limits=self.nthreads):
            if self._embeddings is not None:
                bs = self.batchSize
                n_blocks = int(np.ceil(self.nCells / bs))
                # Two phases: predict must run on the fully fitted model, so it
                # cannot be interleaved with partial_fit. Both passes read the
                # already-materialized embeddings, so no extra data reads occur.
                for start in tqdmbar(
                    range(0, self.nCells, bs),
                    desc="Fitting kmeans",
                    total=n_blocks,
                ):
                    end = min(start + bs, self.nCells)
                    kmeans.partial_fit(self._embeddings[start:end])
                for start in tqdmbar(
                    range(0, self.nCells, bs),
                    desc="Estimating seed partitions",
                    total=n_blocks,
                ):
                    end = min(start + bs, self.nCells)
                    temp.extend(kmeans.predict(self._embeddings[start:end]))
            else:
                for i in self.iter_blocks(msg="Fitting kmeans"):
                    kmeans.partial_fit(self.reducer(i))
                predicted = self.data.map_blocks(
                    lambda _i, s, e: np.asarray(
                        kmeans.predict(self.reducer(self.data._materialize_range(s, e)))
                    ),
                    nthreads=self.nthreads,
                    msg="Estimating seed partitions",
                )
                for part in predicted:
                    temp.extend(part)
        self.clusterLabels = np.array(temp)
        return kmeans
