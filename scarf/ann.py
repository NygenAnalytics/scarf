from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .chunked import ChunkedArray
from .harmony import run_harmony
from .utils import controlled_compute, logger, tqdmbar

__all__ = ["AnnStream", "instantiate_knn_index", "fix_knn_query"]


def instantiate_knn_index(
    space: str,
    dim: int,
    max_elements: int,
    ef_construction: int,
    M: int,
    random_seed: int,
    ef: int,
    num_threads: int,
) -> Any:
    """Create and configure an hnswlib KNN index.

    Args:
        space: Distance metric name accepted by hnswlib (e.g. ``'l2'``, ``'cosine'``).
        dim: Embedding dimensionality.
        max_elements: Maximum number of vectors the index can hold.
        ef_construction: ``ef_construction`` parameter for index building.
        M: ``M`` parameter controlling graph connectivity.
        random_seed: Random seed for index construction.
        ef: ``ef`` search parameter set after index creation.
        num_threads: Number of threads for hnswlib queries.

    Returns:
        Configured hnswlib Index instance.
    """
    import hnswlib

    ann_idx = hnswlib.Index(space=space, dim=dim)
    ann_idx.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )
    ann_idx.set_ef(ef)
    ann_idx.set_num_threads(num_threads)
    return ann_idx


def fix_knn_query(
    indices: np.ndarray,
    distances: np.ndarray,
    ref_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove self-neighbor entries from KNN query results when recall is imperfect.

    Args:
        indices: Neighbor indices from ``knn_query``, shape (n_queries, k).
        distances: Neighbor distances matching ``indices``.
        ref_idx: Global index of each query row (used to detect missing self loops).

    Returns:
        Tuple of (fixed_indices, fixed_distances, n_mis) where ``n_mis`` counts
        queries whose first neighbor was not a self match.
    """
    fixed_ind, fixed_dist = indices.copy()[:, 1:], distances.copy()[:, 1:]
    # Identify positions where first index is not a self loop
    mis_idx = indices[:, 0].reshape(1, -1)[0] != ref_idx
    n_mis = int(mis_idx.sum())
    if n_mis > 0:
        for n, i, j, k in zip(
            np.where(mis_idx)[0], ref_idx[mis_idx], indices[mis_idx], distances[mis_idx]
        ):
            p = np.where(j == i)[0]
            if len(p) > 0:
                # p is the position of self loop. We exclude this position
                p = p[0]
                j = np.array(list(j[:p]) + list(j[p + 1 :]))
                k = np.array(list(k[:p]) + list(k[p + 1 :]))
            else:
                # No self found at all. Poor recall? simply remove the last k neighbour
                j = j[:-1]
                k = k[:-1]
            fixed_ind[n] = j
            fixed_dist[n] = k
    return fixed_ind, fixed_dist, n_mis


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
    ) -> None:
        self.data = data
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
        self.batches = batches
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
            if ann_idx is None:
                self.annIdx = self._fit_ann()
            else:
                self.annIdx = ann_idx
                self.annIdx.set_ef(self.annEf)
                self.annIdx.set_num_threads(self.annThreads)
            self.kmeans = self._fit_kmeans(do_kmeans_fit)

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
        from .storage.zarr_store import profile_prefetch_depth
        from .utils import prefetch_blocks

        blocks = prefetch_blocks(
            self.data.blocks,
            lambda block: controlled_compute(block, self.nthreads),
            max_ahead=profile_prefetch_depth(),
        )
        yield from tqdmbar(blocks, desc=msg, total=self.data.numblocks[0])

    def transform_z(self, a: np.ndarray) -> np.ndarray:
        """Z-score a block using fitted ``mu`` and ``sigma``."""
        return np.asarray((a - self.mu) / self.sigma)

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
        return fix_knn_query(i, d, self_indices)

    def _fit_pca(self, disable_scaling: bool, use_for_pca: np.ndarray) -> None:
        from sklearn.decomposition import IncrementalPCA
        from numpy.linalg import LinAlgError

        assert self.dims is not None
        dims = self.dims

        # We fit 1 extra PC dim than specified and then ignore the last PC.
        self._pca = IncrementalPCA(n_components=dims + 1, batch_size=self.batchSize)
        do_sample_subset = False if use_for_pca.sum() == self.nCells else True
        s, e = 0, 0
        # We store the first block of values here. if such a case arises that we are left with less dims+1 cells to fit
        # then those cells can be added to end_reservoir for fitting. if there are no such cells then end reservoir is
        # just by itself after fitting rest of the cells. If may be the case that the first batch itself has less than
        # dims+1 cells. in that we keep adding cells to carry_over pile until it is big enough.
        end_reservoir: np.ndarray | None = None
        # carry_over store cells that can yet not be added to end_reservoir ot be used for fitting pca directly.
        carry_over: np.ndarray | None = None
        for i in self.iter_blocks(msg="Fitting PCA"):
            if do_sample_subset:
                e = s + i.shape[0]
                i = i[use_for_pca[s:e]]
                s = e
            if disable_scaling is False:
                i = self.transform_z(i)
            if carry_over is not None:
                i = np.vstack((carry_over, i))
                carry_over = None
            if len(i) < (dims + 1):
                carry_over = i
                continue
            if end_reservoir is None:
                end_reservoir = i
                continue
            try:
                self._pca.partial_fit(i, check_input=False)
            except LinAlgError:
                # Add retry counter to make memory consumption doesn't escalate
                carry_over = i
        if carry_over is not None:
            if end_reservoir is not None:
                fit_batch = np.vstack((end_reservoir, carry_over))
            else:
                fit_batch = carry_over
        else:
            assert end_reservoir is not None
            fit_batch = end_reservoir
        try:
            self._pca.partial_fit(fit_batch, check_input=False)
        except LinAlgError:
            logger.warning(
                "{i.shape[0]} samples were not used in PCA fitting due to LinAlgError",
                flush=True,
            )
        self.loadings = self._pca.components_[:-1, :].T

    def _fit_lsi(self, lsi_skip_first: bool, lsi_params: dict[str, Any]) -> None:
        from sklearn.decomposition import TruncatedSVD

        assert self.dims is not None
        dims = self.dims

        reserved = {"n_components", "random_state"}
        for key in list(lsi_params):
            if key in reserved:
                del lsi_params[key]
                logger.warning(
                    f"Provided parameter, {key}, for LSI model will not be used"
                )

        mat = np.vstack(list(self.iter_blocks(msg="Fitting LSI model")))
        svd = TruncatedSVD(
            n_components=dims + 1,
            random_state=self.randState,
            **lsi_params,
        )
        svd.fit(mat)
        components = svd.components_.T
        if lsi_skip_first:
            self.loadings = components[:, 1:]
        else:
            self.loadings = components

    def _fit_ann(self) -> Any:
        def _transform_values() -> np.ndarray:
            pca_array = []
            for _i in self.iter_blocks(msg="Calculating uncorrected latent dimensions"):
                pca_array.append(self.reducer(_i))
            return np.vstack(pca_array).T

        ann_dims = (
            self.dims
            if self.dims is not None and self.dims >= 1
            else self.data.shape[1]
        )

        ann_idx = instantiate_knn_index(
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
                self.harmonizedData = ChunkedArray.from_numpy(
                    run_harmony(_transform_values(), self.batches).T,
                    block_size=self.data.chunksize[0],
                    nthreads=self.nthreads,
                )
            for i in tqdmbar(
                self.harmonizedData.blocks,
                desc="Fitting ANN",
                total=self.harmonizedData.numblocks[0],
            ):
                ann_idx.add_items(controlled_compute(i, self.nthreads))
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
            for i in self.iter_blocks(msg="Fitting kmeans"):
                kmeans.partial_fit(self.reducer(i))
            for i in self.iter_blocks(msg="Estimating seed partitions"):
                temp.extend(kmeans.predict(self.reducer(i)))
        self.clusterLabels = np.array(temp)
        return kmeans
