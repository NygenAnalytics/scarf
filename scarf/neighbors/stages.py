import math
import operator
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from ..embeddings.harmony import HarmonyResult, fit_harmony
from ..matrix import ChunkedArray
from ..utils.logging import logger
from ..utils.process import rss_text
from ..utils.shutdown import shutdown_checkpoint
from .index import fix_knn_query, instantiate_knn_index


class ReductionTransform:
    def __init__(
        self,
        *,
        data: ChunkedArray,
        method: str,
        dims: int | None,
        loadings: np.ndarray | None,
        use_for_pca: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        batch_size: int,
        nthreads: int,
        rand_state: int,
        disable_scaling: bool,
        lsi_skip_first: bool,
        lsi_params: dict[str, Any],
        center: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.method = method
        self.dims = dims
        self.loadings = loadings
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.rand_state = rand_state
        self.pca: Any | None = None
        self.center = center
        disable_reduction = self.dims is not None and self.dims < 1

        if self.method == "pca":
            if self.loadings is None:
                if len(use_for_pca) != self.data.shape[0]:
                    raise ValueError(
                        "ERROR: `use_for_pca` does not have sample length as nCells"
                    )
                if not disable_reduction:
                    with threadpool_limits(limits=self.nthreads):
                        self._fit_pca(disable_scaling, use_for_pca)
            else:
                self.dims = self.loadings.shape[1]
            self._transform = self._pca_transform(
                disable_scaling,
                disable_reduction,
            )
        elif self.method == "lsi":
            if self.loadings is None:
                if not disable_reduction:
                    with threadpool_limits(limits=self.nthreads):
                        self._fit_lsi(lsi_skip_first, lsi_params)
            else:
                self.dims = self.loadings.shape[1]
            self._transform = self._linear_transform(disable_reduction)
        elif self.method == "custom":
            if self.loadings is None:
                raise ValueError("Custom reduction requires loadings")
            self.dims = self.loadings.shape[1]
            self._transform = self._linear_transform(disable_reduction)
        else:
            raise ValueError(f"ERROR: Unknown reduction method: {self.method}")

    def _pca_transform(
        self,
        disable_scaling: bool,
        disable_reduction: bool,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if disable_reduction:
            return (lambda values: values) if disable_scaling else self.transform_z
        if self.center is None:
            raise ValueError("PCA loadings require a fitted center. Re-run run_pca.")
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (self.data.shape[1],) or not np.all(np.isfinite(center)):
            raise ValueError("PCA center must contain one finite value per feature")
        self.center = center
        assert self.loadings is not None
        loadings = self.loadings
        if disable_scaling:
            return lambda values: np.asarray((values - center).dot(loadings))
        return lambda values: np.asarray(
            (self.transform_z(values) - center).dot(loadings)
        )

    def _linear_transform(
        self,
        disable_reduction: bool,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if disable_reduction:
            return lambda values: values
        assert self.loadings is not None
        loadings = self.loadings
        return lambda values: np.asarray(values.dot(loadings))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._transform(values))

    def transform_z(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((values - self.mu) / self.sigma)

    def _fit_pca(
        self,
        disable_scaling: bool,
        use_for_pca: np.ndarray,
    ) -> None:
        from ..embeddings.reduction import fit_incremental_pca

        assert self.dims is not None
        scale = self.transform_z if not disable_scaling else None
        self.loadings, self.pca = fit_incremental_pca(
            self.data,
            dims=self.dims,
            batch_size=self.batch_size,
            use_for_pca=use_for_pca,
            scale=scale,
            nthreads=self.nthreads,
        )
        self.center = np.asarray(self.pca.mean_, dtype=np.float64)

    def _fit_lsi(
        self,
        lsi_skip_first: bool,
        lsi_params: dict[str, Any],
    ) -> None:
        from ..embeddings.reduction import fit_lsi

        assert self.dims is not None
        self.loadings = fit_lsi(
            self.data,
            dims=self.dims,
            skip_first=lsi_skip_first,
            params=lsi_params,
            random_state=self.rand_state,
            nthreads=self.nthreads,
        )


class BatchCorrectionStage:
    def __init__(
        self,
        *,
        stream: "CoordinateSource",
        n_cells: int,
        dims: int,
        batch_size: int,
        batches: pd.DataFrame | None,
        parameters: Mapping[str, Any],
        corrected_data: ChunkedArray | None,
        nthreads: int,
    ) -> None:
        self.stream = stream
        self.n_cells = int(n_cells)
        self.dims = int(dims)
        self.batch_size = int(batch_size)
        self.batches = batches
        self.parameters = dict(parameters)
        self.corrected_data = corrected_data
        self.nthreads = nthreads
        self.result: HarmonyResult | None = None

    def ensure_corrected(self) -> ChunkedArray:
        if self.corrected_data is not None:
            return self.corrected_data
        if self.batches is None:
            raise ValueError("Harmony requires batch metadata")
        uncorrected = np.empty(
            (self.dims, self.n_cells),
            dtype=np.float64,
        )
        start = 0
        for block in self.stream.iter_coordinate_blocks(
            "Loading uncorrected latent dimensions",
        ):
            shutdown_checkpoint()
            values = np.asarray(block)
            del block
            stop = start + int(values.shape[0])
            if values.shape != (stop - start, self.dims) or stop > self.n_cells:
                raise ValueError("Coordinate block has an invalid shape")
            uncorrected[:, start:stop] = values.T
            start = stop
            # A read plan reserves only the blocks in flight, so no block is
            # held while the stream reads the next.
            del values
        if start != self.n_cells:
            raise ValueError(
                f"Coordinate source contains {start} rows, expected {self.n_cells}"
            )
        shutdown_checkpoint()
        with threadpool_limits(limits=self.nthreads):
            self.result = fit_harmony(
                uncorrected,
                self.batches,
                **self.parameters,
            )
        shutdown_checkpoint()
        self.corrected_data = ChunkedArray.from_numpy(
            self.result.corrected.T,
            block_size=self.batch_size,
            nthreads=self.nthreads,
        )
        return self.corrected_data


class CoordinateSource(Protocol):
    def iter_coordinate_blocks(self, message: str) -> Iterator[np.ndarray]: ...


class ChunkedCoordinateStream:
    """Row blocks of a coordinate matrix, read under its memory budget."""

    def __init__(
        self,
        data: ChunkedArray,
        nthreads: int,
        *,
        resident_bytes: int = 0,
    ) -> None:
        self.data = data
        self.nthreads = nthreads
        self.resident_bytes = max(0, int(resident_bytes))

    def iter_coordinate_blocks(self, message: str) -> Iterator[np.ndarray]:
        yield from self.data._stream_blocks(
            nthreads=self.nthreads,
            msg=message,
            prefetch=None,
            row_mask=None,
            resident_bytes=self.resident_bytes,
        )


class AnnIndexStage:
    @staticmethod
    def configure(index: Any, *, ef: int, threads: int) -> Any:
        index.set_ef(ef)
        index.set_num_threads(threads)
        return index

    @staticmethod
    def create(
        *,
        metric: str,
        dims: int,
        n_cells: int,
        ef_construction: int,
        ef: int,
        m: int,
        rand_state: int,
        nthreads: int,
    ) -> Any:
        return instantiate_knn_index(
            metric,
            dims,
            n_cells,
            ef_construction,
            m,
            rand_state,
            ef,
            nthreads,
        )

    @staticmethod
    def populate(index: Any, coordinates: CoordinateSource) -> Any:
        """Add every coordinate block to ``index`` in order."""
        for block in coordinates.iter_coordinate_blocks("Fitting ANN"):
            shutdown_checkpoint()
            index.add_items(block)
            del block
            shutdown_checkpoint()
        return index

    @classmethod
    def fit(
        cls,
        *,
        coordinates: CoordinateSource,
        metric: str,
        dims: int,
        n_cells: int,
        ef_construction: int,
        ef: int,
        m: int,
        rand_state: int,
        nthreads: int,
    ) -> Any:
        index = cls.create(
            metric=metric,
            dims=dims,
            n_cells=n_cells,
            ef_construction=ef_construction,
            ef=ef,
            m=m,
            rand_state=rand_state,
            nthreads=nthreads,
        )
        return cls.populate(index, coordinates)


class NeighborQueryStage:
    def __init__(self, index: Any, k: int, metric: str) -> None:
        self.index = index
        self.k = k
        self.metric = metric

    def _metric_distances(self, distances: np.ndarray) -> np.ndarray:
        values = np.asarray(distances)
        if not np.all(np.isfinite(values)):
            raise ValueError("ANN metric produced non-finite neighbor distances")
        if self.metric in {"l2", "cosine"}:
            if np.any(values < -1e-6):
                raise ValueError("ANN metric produced negative neighbor distances")
            np.maximum(values, 0, out=values)
        if self.metric == "l2":
            np.sqrt(values, out=values)
        if self.metric != "ip" and np.any(values < 0):
            raise ValueError("ANN metric produced negative neighbor distances")
        return values

    def query(
        self,
        values: np.ndarray,
        *,
        self_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        if self_indices is None:
            indices, distances = self.index.knn_query(values, k=self.k)
            return np.asarray(indices), self._metric_distances(distances)
        indices, distances = self.index.knn_query(values, k=self.k + 1)
        fixed_indices, fixed_distances, missed = fix_knn_query(
            indices,
            distances,
            self_indices,
        )
        return (
            fixed_indices,
            self._metric_distances(fixed_distances),
            missed,
        )


@dataclass(frozen=True, slots=True)
class KMeansInitialization:
    model: Any
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class _KMeansSizes:
    """Validated options of a k-means initialization and the sizes they give."""

    batch_size: int
    sampling: float
    clusters: int
    kmeans_batch_size: int
    init_size: int


def _kmeans_sizes(
    n_rows: int,
    *,
    batch_size: int,
    n_clusters: int,
    kmeans_sampling: float,
    kmeans_batch_size: int,
) -> _KMeansSizes:
    if n_rows == 0:
        raise ValueError("K-means initialization requires at least one row")
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        resolved_batch_size = operator.index(batch_size)
    except TypeError:
        raise TypeError("batch_size must be a positive integer") from None
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(kmeans_sampling, bool):
        raise TypeError("kmeans_sampling must be a number")
    try:
        sampling = float(kmeans_sampling)
    except (TypeError, ValueError):
        raise TypeError("kmeans_sampling must be a number") from None
    if not math.isfinite(sampling) or not 0 < sampling <= 1:
        raise ValueError("kmeans_sampling must be greater than 0 and at most 1")
    if isinstance(kmeans_batch_size, bool):
        raise TypeError("kmeans_batch_size must be a positive integer")
    try:
        requested_kmeans_batch_size = operator.index(kmeans_batch_size)
    except TypeError:
        raise TypeError("kmeans_batch_size must be a positive integer") from None
    if requested_kmeans_batch_size < 1:
        raise ValueError("kmeans_batch_size must be a positive integer")
    clusters = min(max(n_clusters, 2), n_rows)
    if clusters < 2:
        raise ValueError("K-means initialization requires at least two rows")
    return _KMeansSizes(
        batch_size=resolved_batch_size,
        sampling=sampling,
        clusters=clusters,
        kmeans_batch_size=min(n_rows, max(requested_kmeans_batch_size, clusters)),
        init_size=min(n_rows, max(clusters, math.ceil(n_rows * sampling))),
    )


@dataclass(frozen=True, slots=True)
class KMeansFitMemory:
    """Bytes ``KMeansInitializationStage.fit`` holds besides its stream's blocks."""

    peakBytes: int
    streamResidentBytes: int


# Rows of distances to every centroid that each OpenMP thread of
# scikit-learn's chunked k-means loops holds.
_KMEANS_THREAD_CHUNK_ROWS = 256
# Python objects a k-means fit holds whatever the data size, such as the
# estimator's state and thread-limit records, measured below 80 KiB.
_KMEANS_OBJECT_BYTES = 128 * 1024


def _kmeans_plusplus_bytes(rows: int, dims: int, clusters: int, itemsize: int) -> int:
    """Bytes scikit-learn's k-means++ holds to seed from ``rows`` points.

    Besides the points, it holds their squared norms and weights, the
    potentials it draws candidates from and float64 seeding probabilities,
    the distances of ``2 + log(clusters)`` candidates to every point for the
    current and the previous candidates (with a product temporary unless the
    points are float32), and the seeds. It computes float32 distances in
    float64 chunks of at most ``sqrt(M)`` points, where ``M`` is a tenth of
    the elements involved and at least ``10 * 2**17``.
    """
    trials = 2 + int(math.log(clusters))
    vectors = rows * (4 * itemsize + 16)
    distances = (2 if itemsize == 4 else 3) * trials * rows * itemsize
    seeds = clusters * (dims * itemsize + 8)
    chunks = 0
    if itemsize == 4:
        elements = max(((trials + rows) * dims + trials * rows) // 10, 10 * 2**17)
        chunk_rows = min(rows, math.isqrt(elements) + 1)
        chunks = 8 * ((chunk_rows + trials) * (dims + 1) + 3 * trials * chunk_rows)
    return vectors + distances + seeds + chunks


def kmeans_fit_memory(
    *,
    n_rows: int,
    dims: int,
    dtype: Any,
    batch_size: int,
    n_clusters: int,
    block_rows: int,
    nthreads: int,
    kmeans_sampling: float = 0.1,
    kmeans_batch_size: int = 10_000,
) -> KMeansFitMemory:
    """Estimate the bytes ``KMeansInitializationStage.fit`` holds."""
    sizes = _kmeans_sizes(
        n_rows,
        batch_size=batch_size,
        n_clusters=n_clusters,
        kmeans_sampling=kmeans_sampling,
        kmeans_batch_size=kmeans_batch_size,
    )
    coordinate_dtype = np.dtype(dtype)
    itemsize = coordinate_dtype.itemsize
    # scikit-learn fits float32 and float64 values as they are and copies
    # values of any other floating dtype to float64.
    native = coordinate_dtype in (np.dtype(np.float32), np.dtype(np.float64))
    fit_itemsize = itemsize if native else 8
    copy_itemsize = 0 if native else 8
    clusters = sizes.clusters
    update_rows = sizes.kmeans_batch_size
    init_size = sizes.init_size
    block = min(max(1, int(block_rows)), n_rows)
    # The seeds and the centroids updated from them, their counts, and the row
    # indices of the seeds.
    model = clusters * dims * (fit_itemsize + max(itemsize, fit_itemsize)) + (
        clusters * (fit_itemsize + 8)
    )
    # Each thread holds the distances of up to 256 rows to every centroid,
    # its own centroid sums, and the row indices of one update.
    threads = max(1, int(nthreads)) * (
        clusters * (_KMEANS_THREAD_CHUNK_ROWS + dims + 1) * fit_itemsize
        + update_rows * 4
    )
    if sizes.batch_size >= n_rows:
        matrix = n_rows * dims * (itemsize + copy_itemsize)
        # Sample weights and squared norms of every row, and the validation
        # sample with its weights, row indices, and labels.
        held = 2 * n_rows * fit_itemsize + init_size * (
            dims * fit_itemsize + fit_itemsize + 12
        )
        seed = init_size * (dims * fit_itemsize + 8) + _kmeans_plusplus_bytes(
            init_size, dims, clusters, fit_itemsize
        )
        # Each mini-batch draw takes the cumulative sums of float64 row
        # probabilities, converted from float32 weights, while the row
        # indices of the previous mini-batch are still held.
        draw = n_rows * (16 if fit_itemsize == 4 else 8) + update_rows * 32
        # A mini-batch holds its rows, their indices, labels, and a
        # permutation, and the rows that replace rarely used centroids.
        minibatch = update_rows * (dims * fit_itemsize + 20) + min(
            clusters, update_rows
        ) * (dims * fit_itemsize + 9)
        steps = (
            n_rows * fit_itemsize + update_rows * fit_itemsize + max(draw, minibatch)
        )
        labelling = n_rows * (fit_itemsize + 4) + update_rows * (fit_itemsize + 8)
        return KMeansFitMemory(
            peakBytes=matrix
            + held
            + max(seed, steps, labelling)
            + model
            + threads
            + _KMEANS_OBJECT_BYTES,
            streamResidentBytes=_KMEANS_OBJECT_BYTES
            + (0 if block >= n_rows else n_rows * dims * itemsize),
        )
    # The seeding sample and its sorted row indices, beside the unsorted
    # indices or the indices of one block's sampled rows.
    sampling = init_size * (dims * itemsize + 16)
    # k-means++ checks a float64 copy of a sample of another dtype, then seeds
    # from the sample as it is, which float64 seeding bounds.
    seed = init_size * (dims * itemsize + 8) + max(
        init_size * dims * copy_itemsize,
        _kmeans_plusplus_bytes(
            init_size, dims, clusters, itemsize if native else max(8, itemsize)
        ),
    )
    # The first update seeds scikit-learn's state from a subsample of
    # ``init_size`` of its rows when it holds more.
    subsample = (
        init_size * (dims * fit_itemsize + 2 * fit_itemsize + 8)
        if init_size < update_rows
        else 0
    )
    updating = (
        update_rows * dims * itemsize
        + update_rows * (dims * copy_itemsize + 2 * fit_itemsize + 8)
        + subsample
        + model
        + threads
    )
    labelling = (
        n_rows * 4
        + update_rows * 4
        + block * (dims * copy_itemsize + fit_itemsize + 4)
        + model
        + threads
    )
    resident = max(sampling, updating, labelling) + _KMEANS_OBJECT_BYTES
    return KMeansFitMemory(
        peakBytes=max(resident, seed + _KMEANS_OBJECT_BYTES),
        streamResidentBytes=resident,
    )


class KMeansInitializationStage:
    @staticmethod
    def fit(
        *,
        stream: CoordinateSource,
        n_rows: int,
        batch_size: int,
        n_clusters: int,
        rand_state: int,
        nthreads: int,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
    ) -> KMeansInitialization:
        """Fit mini-batch k-means centroids and label every row."""
        sizes = _kmeans_sizes(
            n_rows,
            batch_size=batch_size,
            n_clusters=n_clusters,
            kmeans_sampling=kmeans_sampling,
            kmeans_batch_size=kmeans_batch_size,
        )
        from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
        from sklearn.utils.random import sample_without_replacement

        resolved_batch_size = sizes.batch_size
        in_memory = resolved_batch_size >= n_rows
        resolved_kmeans_sampling = sizes.sampling
        effective_clusters = sizes.clusters
        effective_kmeans_batch_size = sizes.kmeans_batch_size
        init_size = sizes.init_size

        def make_model(
            *,
            init: str | np.ndarray = "k-means++",
            reassignment_ratio: float = 0.01,
        ) -> Any:
            return MiniBatchKMeans(
                n_clusters=effective_clusters,
                random_state=rand_state,
                batch_size=effective_kmeans_batch_size,
                init_size=init_size,
                init=init,
                n_init=1,
                reassignment_ratio=reassignment_ratio,
            )

        def timed_blocks(
            message: str,
        ) -> Generator[tuple[int, np.ndarray, float, float]]:
            # A read plan reserves only the blocks in flight, so neither this
            # generator nor its consumer holds a block while the next is read.
            blocks = iter(stream.iter_coordinate_blocks(message))
            block_idx = 0
            while True:
                shutdown_checkpoint()
                wall_started = time.perf_counter()
                cpu_started = time.process_time()
                try:
                    block = next(blocks)
                except StopIteration:
                    return
                block_idx += 1
                yield (
                    block_idx,
                    block,
                    time.perf_counter() - wall_started,
                    time.process_time() - cpu_started,
                )
                del block

        with threadpool_limits(limits=nthreads):
            pools = sorted(
                (
                    str(pool.get("user_api")),
                    str(pool.get("internal_api")),
                    int(pool.get("num_threads", 0)),
                )
                for pool in threadpool_info()
            )
            logger.debug(
                f"KMeans initialization plan: rows={n_rows} "
                f"batchSize={resolved_batch_size} "
                f"fit={'in memory' if in_memory else 'streamed'} "
                f"clusters={effective_clusters} "
                f"samplingFraction={resolved_kmeans_sampling:.4f} "
                f"initSize={init_size} "
                f"kmeansBatchSize={effective_kmeans_batch_size} "
                f"requestedThreads={nthreads} threadPools={pools}"
            )
            coordinate_blocks = timed_blocks("Loading kmeans coordinates")
            try:
                block_idx, block, read_seconds, read_cpu_seconds = next(
                    coordinate_blocks
                )
            except StopIteration:
                raise ValueError(
                    "K-means initialization coordinate source is empty"
                ) from None
            if block.ndim != 2:
                raise ValueError("K-means coordinate blocks must be two-dimensional")
            if in_memory:
                if block.shape[0] == n_rows:
                    values = block
                    try:
                        next(coordinate_blocks)
                    except StopIteration:
                        pass
                    else:
                        coordinate_blocks.close()
                        raise ValueError(
                            "K-means coordinate source yielded rows after a "
                            "complete block"
                        )
                else:
                    # The stream splits the rows into several blocks; gather
                    # them, so the fit sees the same rows as for one block.
                    values = np.empty((n_rows, int(block.shape[1])), dtype=block.dtype)
                    gathered_rows = 0
                    while True:
                        shutdown_checkpoint()
                        if (
                            block.ndim != 2
                            or block.shape[1] != values.shape[1]
                            or block.dtype != values.dtype
                        ):
                            raise ValueError(
                                "K-means coordinate block dimensions changed"
                            )
                        block_stop = gathered_rows + int(block.shape[0])
                        if block_stop > n_rows:
                            raise ValueError(
                                "K-means coordinate source has too many rows"
                            )
                        values[gathered_rows:block_stop] = block
                        gathered_rows = block_stop
                        del block
                        try:
                            block_idx, block, block_seconds, block_cpu_seconds = next(
                                coordinate_blocks
                            )
                        except StopIteration:
                            break
                        read_seconds += block_seconds
                        read_cpu_seconds += block_cpu_seconds
                    if gathered_rows != n_rows:
                        raise ValueError(
                            f"K-means coordinate source contains {gathered_rows} "
                            f"rows, expected {n_rows}"
                        )
                model = make_model()
                compute_started = time.perf_counter()
                compute_cpu_started = time.process_time()
                model.fit(values)
                compute_seconds = time.perf_counter() - compute_started
                compute_cpu_seconds = time.process_time() - compute_cpu_started
                logger.opt(lazy=True).debug(
                    f"KMeans in-memory minibatch fit: blocks={block_idx} "
                    f"rows={values.shape[0]} "
                    f"read={read_seconds:.3f}s readCpu={read_cpu_seconds:.3f}s "
                    f"readCores={read_cpu_seconds / max(read_seconds, 1e-12):.2f} "
                    f"compute={compute_seconds:.3f}s "
                    f"computeCpu={compute_cpu_seconds:.3f}s "
                    f"computeCores="
                    f"{compute_cpu_seconds / max(compute_seconds, 1e-12):.2f} "
                    f"steps={model.n_steps_} iterations={model.n_iter_} "
                    f"inertiaPerRow={float(model.inertia_) / n_rows:.6f} "
                    "rss={rss}",
                    rss=rss_text,
                )
                return KMeansInitialization(
                    model=model,
                    labels=np.asarray(model.labels_, dtype=np.uint32),
                )

            coordinate_dims = int(block.shape[1])
            coordinate_dtype = block.dtype
            sample_indices = np.sort(
                sample_without_replacement(
                    n_rows,
                    init_size,
                    method="reservoir_sampling",
                    random_state=rand_state,
                )
            )
            sample = np.empty((init_size, coordinate_dims), dtype=block.dtype)
            rows_seen = 0
            sample_blocks = 0
            sample_read_seconds = 0.0
            sample_read_cpu_seconds = 0.0
            sample_started = time.perf_counter()
            sample_cpu_started = time.process_time()
            while True:
                shutdown_checkpoint()
                if (
                    block.ndim != 2
                    or int(block.shape[1]) != coordinate_dims
                    or block.dtype != coordinate_dtype
                ):
                    raise ValueError("K-means coordinate block dimensions changed")
                block_rows = int(block.shape[0])
                block_stop = rows_seen + block_rows
                if block_stop > n_rows:
                    raise ValueError("K-means coordinate source has too many rows")
                sample_start = int(
                    np.searchsorted(sample_indices, rows_seen, side="left")
                )
                sample_stop = int(
                    np.searchsorted(sample_indices, block_stop, side="left")
                )
                local_indices = sample_indices[sample_start:sample_stop] - rows_seen
                # The sampled rows are copied straight into the sample, without
                # a temporary; the indices lie within the block by construction,
                # so clipping never changes them.
                np.take(
                    block,
                    local_indices,
                    axis=0,
                    out=sample[sample_start:sample_stop],
                    mode="clip",
                )
                del local_indices, block
                rows_seen = block_stop
                sample_blocks += 1
                sample_read_seconds += read_seconds
                sample_read_cpu_seconds += read_cpu_seconds
                try:
                    block_idx, block, read_seconds, read_cpu_seconds = next(
                        coordinate_blocks
                    )
                except StopIteration:
                    break
            if rows_seen != n_rows:
                raise ValueError(
                    f"K-means coordinate source contains {rows_seen} rows, "
                    f"expected {n_rows}"
                )
            sample_seconds = time.perf_counter() - sample_started
            sample_cpu_seconds = time.process_time() - sample_cpu_started
            logger.opt(lazy=True).debug(
                f"KMeans sampling pass: blocks={sample_blocks} rows={rows_seen} "
                f"sampleRows={init_size} wall={sample_seconds:.3f}s "
                f"cpu={sample_cpu_seconds:.3f}s "
                f"effectiveCores="
                f"{sample_cpu_seconds / max(sample_seconds, 1e-12):.2f} "
                f"read={sample_read_seconds:.3f}s "
                f"readCpu={sample_read_cpu_seconds:.3f}s "
                "rss={rss}",
                rss=rss_text,
            )

            seed_started = time.perf_counter()
            seed_cpu_started = time.process_time()
            initial_centers, _ = kmeans_plusplus(
                sample,
                n_clusters=effective_clusters,
                random_state=rand_state,
            )
            seed_seconds = time.perf_counter() - seed_started
            seed_cpu_seconds = time.process_time() - seed_cpu_started
            logger.opt(lazy=True).debug(
                f"KMeans centroid seeding: sampleRows={init_size} "
                f"compute={seed_seconds:.3f}s cpu={seed_cpu_seconds:.3f}s "
                f"effectiveCores="
                f"{seed_cpu_seconds / max(seed_seconds, 1e-12):.2f} "
                "rss={rss}",
                rss=rss_text,
            )
            del sample, sample_indices

            # Updates follow storage order, where cells are often grouped by
            # sample. Moving rarely used centroids onto the current batch would
            # pull centroids of later groups onto earlier ones.
            model = make_model(
                init=np.asarray(initial_centers),
                reassignment_ratio=0.0,
            )
            update_buffer = np.empty(
                (effective_kmeans_batch_size, coordinate_dims),
                dtype=coordinate_dtype,
            )
            buffered_rows = 0
            fitted_rows = 0
            fit_blocks = 0
            update_count = 0
            fit_read_seconds = 0.0
            fit_read_cpu_seconds = 0.0
            fit_compute_seconds = 0.0
            fit_compute_cpu_seconds = 0.0
            for _, block, read_seconds, read_cpu_seconds in timed_blocks(
                "Fitting kmeans"
            ):
                shutdown_checkpoint()
                if (
                    block.ndim != 2
                    or int(block.shape[1]) != coordinate_dims
                    or block.dtype != coordinate_dtype
                ):
                    raise ValueError("K-means coordinate block dimensions changed")
                block_rows = int(block.shape[0])
                fitted_rows += block_rows
                if fitted_rows > n_rows:
                    raise ValueError("K-means coordinate source has too many rows")
                fit_blocks += 1
                fit_read_seconds += read_seconds
                fit_read_cpu_seconds += read_cpu_seconds
                compute_started = time.perf_counter()
                compute_cpu_started = time.process_time()
                block_offset = 0
                while block_offset < block_rows:
                    shutdown_checkpoint()
                    rows_to_copy = min(
                        effective_kmeans_batch_size - buffered_rows,
                        block_rows - block_offset,
                    )
                    update_buffer[buffered_rows : buffered_rows + rows_to_copy] = block[
                        block_offset : block_offset + rows_to_copy
                    ]
                    buffered_rows += rows_to_copy
                    block_offset += rows_to_copy
                    if buffered_rows == effective_kmeans_batch_size:
                        model.partial_fit(update_buffer)
                        update_count += 1
                        buffered_rows = 0
                del block
                fit_compute_seconds += time.perf_counter() - compute_started
                fit_compute_cpu_seconds += time.process_time() - compute_cpu_started
            if fitted_rows != n_rows:
                raise ValueError(
                    f"K-means coordinate source contains {fitted_rows} rows, "
                    f"expected {n_rows}"
                )
            if buffered_rows:
                compute_started = time.perf_counter()
                compute_cpu_started = time.process_time()
                model.partial_fit(update_buffer[:buffered_rows])
                update_count += 1
                fit_compute_seconds += time.perf_counter() - compute_started
                fit_compute_cpu_seconds += time.process_time() - compute_cpu_started
            del update_buffer
            logger.opt(lazy=True).debug(
                f"KMeans streaming fit: blocks={fit_blocks} rows={fitted_rows} "
                f"updates={update_count} read={fit_read_seconds:.3f}s "
                f"readCpu={fit_read_cpu_seconds:.3f}s "
                f"compute={fit_compute_seconds:.3f}s "
                f"computeCpu={fit_compute_cpu_seconds:.3f}s "
                f"computeCores="
                f"{fit_compute_cpu_seconds / max(fit_compute_seconds, 1e-12):.2f} "
                "rss={rss}",
                rss=rss_text,
            )

            labels = np.empty(n_rows, dtype=np.uint32)
            predicted_rows = 0
            predict_blocks = 0
            predict_read_seconds = 0.0
            predict_read_cpu_seconds = 0.0
            predict_compute_seconds = 0.0
            predict_compute_cpu_seconds = 0.0
            for _, block, read_seconds, read_cpu_seconds in timed_blocks(
                "Estimating seed partitions"
            ):
                shutdown_checkpoint()
                if (
                    block.ndim != 2
                    or int(block.shape[1]) != coordinate_dims
                    or block.dtype != coordinate_dtype
                ):
                    raise ValueError("K-means coordinate block dimensions changed")
                block_rows = int(block.shape[0])
                block_stop = predicted_rows + block_rows
                if block_stop > n_rows:
                    raise ValueError("K-means coordinate source has too many rows")
                predict_blocks += 1
                predict_read_seconds += read_seconds
                predict_read_cpu_seconds += read_cpu_seconds
                compute_started = time.perf_counter()
                compute_cpu_started = time.process_time()
                labels[predicted_rows:block_stop] = model.predict(block)
                del block
                predict_compute_seconds += time.perf_counter() - compute_started
                predict_compute_cpu_seconds += time.process_time() - compute_cpu_started
                predicted_rows = block_stop
            if predicted_rows != n_rows:
                raise ValueError(
                    f"K-means coordinate source contains {predicted_rows} rows, "
                    f"expected {n_rows}"
                )
            logger.opt(lazy=True).debug(
                f"KMeans prediction pass: blocks={predict_blocks} "
                f"rows={predicted_rows} read={predict_read_seconds:.3f}s "
                f"readCpu={predict_read_cpu_seconds:.3f}s "
                f"compute={predict_compute_seconds:.3f}s "
                f"computeCpu={predict_compute_cpu_seconds:.3f}s "
                f"computeCores="
                f"{predict_compute_cpu_seconds / max(predict_compute_seconds, 1e-12):.2f} "
                "rss={rss}",
                rss=rss_text,
            )
        return KMeansInitialization(
            model=model,
            labels=labels,
        )
