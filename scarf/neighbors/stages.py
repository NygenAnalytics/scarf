from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..embeddings.harmony import HarmonyResult, fit_harmony
from ..matrix import ChunkedArray
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.progress import tqdmbar
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
        for block in tqdmbar(
            self.data.blocks,
            desc=message,
            total=self.data.numblocks[0],
        ):
            yield np.asarray(controlled_compute(block, self.nthreads))


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
        ann_threads: int,
    ) -> Any:
        return instantiate_knn_index(
            metric,
            dims,
            n_cells,
            ef_construction,
            m,
            rand_state,
            ef,
            ann_threads,
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
        ann_threads: int,
    ) -> Any:
        index = cls.create(
            metric=metric,
            dims=dims,
            n_cells=n_cells,
            ef_construction=ef_construction,
            ef=ef,
            m=m,
            rand_state=rand_state,
            ann_threads=ann_threads,
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
        from sklearn.cluster import MiniBatchKMeans

        effective_clusters = min(
            max(n_clusters, 2),
            batch_size,
            n_rows,
        )
        if effective_clusters < 2:
            raise ValueError(
                "K-means initialization requires at least two rows per batch"
            )
        model = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=rand_state,
            batch_size=batch_size,
            n_init=3,
        )
        labels: list[int] = []
        with threadpool_limits(limits=nthreads):
            for block in stream.iter_coordinate_blocks("Fitting kmeans"):
                model.partial_fit(block)
            for block in stream.iter_coordinate_blocks(
                "Estimating seed partitions",
            ):
                labels.extend(model.predict(block))
        return KMeansInitialization(
            model=model,
            labels=np.asarray(labels, dtype=np.uint32),
        )
