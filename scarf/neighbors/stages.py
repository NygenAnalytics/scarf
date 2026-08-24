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
from ..utils.process import process_rss_mb
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
        self.feature_scaling = not disable_scaling
        self.pca: Any | None = None
        disable_reduction = self.dims is not None and self.dims < 1

        if self.method == "pca":
            if self.loadings is None or len(self.loadings) == 0:
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
            if self.loadings is None or len(self.loadings) == 0:
                if not disable_reduction:
                    with threadpool_limits(limits=self.nthreads):
                        self._fit_lsi(lsi_skip_first, lsi_params)
            else:
                self.dims = self.loadings.shape[1]
            self._transform = self._linear_transform(disable_reduction)
        elif self.method == "custom":
            if self.loadings is None or len(self.loadings) == 0:
                logger.warning("No loadings provided for manual dimension reduction")
            else:
                self.dims = self.loadings.shape[1]
            self._transform = self._linear_transform(disable_reduction)
        else:
            raise ValueError(f"ERROR: Unknown reduction method: {self.method}")

    def _pca_transform(
        self,
        disable_scaling: bool,
        disable_reduction: bool,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if disable_scaling:
            if disable_reduction:
                return lambda values: values
            assert self.loadings is not None
            loadings = self.loadings
            return lambda values: np.asarray(values.dot(loadings))
        if disable_reduction:
            return self.transform_z
        assert self.loadings is not None
        loadings = self.loadings
        return lambda values: np.asarray(self.transform_z(values).dot(loadings))

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


class LazyTransformStream:
    def __init__(
        self,
        *,
        data: ChunkedArray,
        transform: Callable[[np.ndarray], np.ndarray],
        nthreads: int,
        batch_size: int,
    ) -> None:
        self.data = data
        self.transform = transform
        self.nthreads = nthreads
        self.batch_size = batch_size

    def iter_raw(self, message: str = "") -> Iterator[np.ndarray]:
        yield from self.data.stream_blocks(nthreads=self.nthreads, msg=message)

    def iter_transformed(self, message: str = "") -> Iterator[np.ndarray]:
        for block in self.iter_raw(message):
            yield self.transform(block)

    def iter_coordinate_blocks(self, message: str) -> Iterator[np.ndarray]:
        yield from self.iter_transformed(message)


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
            values = np.asarray(block)
            stop = start + int(values.shape[0])
            if values.shape != (stop - start, self.dims) or stop > self.n_cells:
                raise ValueError("Coordinate block has an invalid shape")
            uncorrected[:, start:stop] = values.T
            start = stop
        if start != self.n_cells:
            raise ValueError(
                f"Coordinate source contains {start} rows, expected {self.n_cells}"
            )
        with threadpool_limits(limits=self.nthreads):
            self.result = fit_harmony(
                uncorrected,
                self.batches,
                **self.parameters,
            )
        self.corrected_data = ChunkedArray.from_numpy(
            self.result.corrected.T,
            block_size=self.batch_size,
            nthreads=self.nthreads,
        )
        return self.corrected_data


class CoordinateSource(Protocol):
    def iter_coordinate_blocks(self, message: str) -> Iterator[np.ndarray]: ...


class ChunkedCoordinateStream:
    def __init__(self, data: ChunkedArray, nthreads: int) -> None:
        self.data = data
        self.nthreads = nthreads

    def iter_coordinate_blocks(self, message: str) -> Iterator[np.ndarray]:
        yield from self.data.stream_blocks(
            nthreads=self.nthreads,
            msg=message,
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
        for block in coordinates.iter_coordinate_blocks("Fitting ANN"):
            index.add_items(block)
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
        k: int | None = None,
        self_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int]:
        use_k = self.k if k is None else k
        if self_indices is None:
            indices, distances = self.index.knn_query(values, k=use_k)
            return np.asarray(indices), self._metric_distances(distances)
        indices, distances = self.index.knn_query(values, k=use_k + 1)
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
    model: Any | None
    labels: np.ndarray


class KMeansInitializationStage:
    @staticmethod
    def fit(
        *,
        stream: CoordinateSource,
        n_rows: int | None = None,
        batch_size: int | None = None,
        n_clusters: int,
        rand_state: int,
        nthreads: int,
        enabled: bool,
        kmeans_sampling: float = 0.1,
        kmeans_batch_size: int = 10_000,
    ) -> KMeansInitialization:
        if n_rows is None:
            data = getattr(stream, "data", None)
            if data is None:
                raise ValueError("n_rows is required for this coordinate source")
            n_rows = int(data.shape[0])
        if batch_size is None:
            raw_batch_size = getattr(stream, "batch_size", None)
            if raw_batch_size is None:
                raise ValueError("batch_size is required for this coordinate source")
            batch_size = int(raw_batch_size)
        if not enabled:
            return KMeansInitialization(
                model=None,
                labels=np.repeat(-1, n_rows),
            )
        if n_rows == 0:
            raise ValueError("K-means initialization requires at least one row")
        if isinstance(kmeans_sampling, bool):
            raise TypeError("kmeans_sampling must be a number")
        try:
            resolved_kmeans_sampling = float(kmeans_sampling)
        except (TypeError, ValueError):
            raise TypeError("kmeans_sampling must be a number") from None
        if (
            not math.isfinite(resolved_kmeans_sampling)
            or not 0 < resolved_kmeans_sampling <= 1
        ):
            raise ValueError("kmeans_sampling must be greater than 0 and at most 1")
        if isinstance(kmeans_batch_size, bool):
            raise TypeError("kmeans_batch_size must be a positive integer")
        try:
            requested_kmeans_batch_size = operator.index(kmeans_batch_size)
        except TypeError:
            raise TypeError("kmeans_batch_size must be a positive integer") from None
        if requested_kmeans_batch_size < 1:
            raise ValueError("kmeans_batch_size must be a positive integer")
        from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
        from sklearn.utils.random import sample_without_replacement

        effective_clusters = min(
            max(n_clusters, 2),
            n_rows,
        )
        if effective_clusters < 2:
            raise ValueError("K-means initialization requires at least two rows")
        effective_kmeans_batch_size = min(
            n_rows,
            max(requested_kmeans_batch_size, effective_clusters),
        )
        init_size = min(
            n_rows,
            max(effective_clusters, math.ceil(n_rows * resolved_kmeans_sampling)),
        )

        def make_model(*, init: str | np.ndarray = "k-means++") -> Any:
            return MiniBatchKMeans(
                n_clusters=effective_clusters,
                random_state=rand_state,
                batch_size=effective_kmeans_batch_size,
                init_size=init_size,
                init=init,
                n_init=1,
            )

        def timed_blocks(
            message: str,
        ) -> Generator[tuple[int, np.ndarray, float, float]]:
            blocks = iter(stream.iter_coordinate_blocks(message))
            block_idx = 0
            while True:
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
                f"readBatchSize={batch_size} clusters={effective_clusters} "
                f"samplingFraction={resolved_kmeans_sampling:.4f} "
                f"initSize={init_size} "
                f"kmeansBatchSize={effective_kmeans_batch_size} "
                f"requestedThreads={nthreads} threadPools={pools}"
            )
            coordinate_blocks = timed_blocks("Loading kmeans coordinates")
            try:
                first_block = next(coordinate_blocks)
            except StopIteration:
                raise ValueError(
                    "K-means initialization coordinate source is empty"
                ) from None
            block_idx, block, read_seconds, read_cpu_seconds = first_block
            if block.ndim != 2:
                raise ValueError("K-means coordinate blocks must be two-dimensional")
            if block.shape[0] == n_rows:
                try:
                    next(coordinate_blocks)
                except StopIteration:
                    pass
                else:
                    coordinate_blocks.close()
                    raise ValueError(
                        "K-means coordinate source yielded rows after a complete block"
                    )
                model = make_model()
                compute_started = time.perf_counter()
                compute_cpu_started = time.process_time()
                model.fit(block)
                compute_seconds = time.perf_counter() - compute_started
                compute_cpu_seconds = time.process_time() - compute_cpu_started
                logger.debug(
                    f"KMeans minibatch fit block {block_idx}: rows={block.shape[0]} "
                    f"read={read_seconds:.3f}s readCpu={read_cpu_seconds:.3f}s "
                    f"readCores={read_cpu_seconds / max(read_seconds, 1e-12):.2f} "
                    f"compute={compute_seconds:.3f}s "
                    f"computeCpu={compute_cpu_seconds:.3f}s "
                    f"computeCores="
                    f"{compute_cpu_seconds / max(compute_seconds, 1e-12):.2f} "
                    f"steps={model.n_steps_} iterations={model.n_iter_} "
                    f"inertiaPerRow={float(model.inertia_) / n_rows:.6f} "
                    f"rss={process_rss_mb():.0f} MiB"
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
                sample[sample_start:sample_stop] = block[local_indices]
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
            logger.debug(
                f"KMeans sampling pass: blocks={sample_blocks} rows={rows_seen} "
                f"sampleRows={init_size} wall={sample_seconds:.3f}s "
                f"cpu={sample_cpu_seconds:.3f}s "
                f"effectiveCores="
                f"{sample_cpu_seconds / max(sample_seconds, 1e-12):.2f} "
                f"read={sample_read_seconds:.3f}s "
                f"readCpu={sample_read_cpu_seconds:.3f}s "
                f"rss={process_rss_mb():.0f} MiB"
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
            logger.debug(
                f"KMeans centroid seeding: sampleRows={init_size} "
                f"compute={seed_seconds:.3f}s cpu={seed_cpu_seconds:.3f}s "
                f"effectiveCores="
                f"{seed_cpu_seconds / max(seed_seconds, 1e-12):.2f} "
                f"rss={process_rss_mb():.0f} MiB"
            )
            del sample, sample_indices

            model = make_model(init=np.asarray(initial_centers))
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
            logger.debug(
                f"KMeans streaming fit: blocks={fit_blocks} rows={fitted_rows} "
                f"updates={update_count} read={fit_read_seconds:.3f}s "
                f"readCpu={fit_read_cpu_seconds:.3f}s "
                f"compute={fit_compute_seconds:.3f}s "
                f"computeCpu={fit_compute_cpu_seconds:.3f}s "
                f"computeCores="
                f"{fit_compute_cpu_seconds / max(fit_compute_seconds, 1e-12):.2f} "
                f"rss={process_rss_mb():.0f} MiB"
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
                predict_compute_seconds += time.perf_counter() - compute_started
                predict_compute_cpu_seconds += time.process_time() - compute_cpu_started
                predicted_rows = block_stop
            if predicted_rows != n_rows:
                raise ValueError(
                    f"K-means coordinate source contains {predicted_rows} rows, "
                    f"expected {n_rows}"
                )
            logger.debug(
                f"KMeans prediction pass: blocks={predict_blocks} "
                f"rows={predicted_rows} read={predict_read_seconds:.3f}s "
                f"readCpu={predict_read_cpu_seconds:.3f}s "
                f"compute={predict_compute_seconds:.3f}s "
                f"computeCpu={predict_compute_cpu_seconds:.3f}s "
                f"computeCores="
                f"{predict_compute_cpu_seconds / max(predict_compute_seconds, 1e-12):.2f} "
                f"rss={process_rss_mb():.0f} MiB"
            )
        return KMeansInitialization(
            model=model,
            labels=labels,
        )
