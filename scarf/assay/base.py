from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, vstack

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import array_digest
from ..utils.compute import controlled_compute, compute_with_progress
from ..utils.logging import logger
from .normalization import NormMethod, norm_dummy, norm_lib_size

type PercentFeatures = dict[str, str]

_DEFER_FEATURE_PROPS: ContextVar[bool] = ContextVar(
    "scarf_defer_feature_props",
    default=False,
)


@contextmanager
def _defer_feature_props() -> Generator[None, None, None]:
    token = _DEFER_FEATURE_PROPS.set(True)
    try:
        yield
    finally:
        _DEFER_FEATURE_PROPS.reset(token)


class Assay:
    """A generic Assay class that contains methods to calculate feature level
    statistics and stream normalized values for downstream computation.

    Args:
        z (zarr.Group): Zarr hierarchy where raw data is located
        workspace: Workspace name when assays live under ``matrices/`` (None for legacy layout)
        name (str): A label/name for assay.
        cell_data: Metadata class object for the cell attributes.
        nthreads: number of threads to use for parallel computations

    Attributes:
        name: A label for the assay instance
        z: Zarr group that contains the assay
        cells: A Metadata class object for cell attributes
        nthreads: number of threads to use for computations
        rawData: chunked array containing the raw data
        feats: a MetaData class object for feature attributes
        attrs: Zarr attributes for the zarr group of the assay
        normMethod: normalization method to use.
        sf: scaling factor for doing library-size normalization
    """

    def __init__(
        self,
        z: zarr.Group,
        workspace: str | None,
        name: str,  # FIXME change to assay_name
        cell_data: MetaData,
        nthreads: int,
        matrix_root: zarr.Group | None = None,
        resources: ResourceBudget | None = None,
        storageIo: Any | None = None,
    ) -> None:
        self.name = name
        self.cells = cell_data
        self.resources = resources or resolve_budget(workers=nthreads)
        self.nthreads = self.resources.workers
        self.storageIo = storageIo
        matrix_root = z if matrix_root is None else matrix_root
        if workspace is None:
            self._artifact_root = z
            counts_path = f"{name}/counts"
            counts_t_path = f"{name}/countsT"
            matrix_group = as_zarr_group(matrix_root[name], name=name)
            self.rawData = ChunkedArray(
                as_zarr_array(matrix_root[counts_path], name=counts_path),
                nthreads=self.nthreads,
                resources=self.resources,
            )
            self.rawData._io = self.storageIo
            self.feats = MetaData(z[f"{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[name], name=name)
        else:
            self._artifact_root = as_zarr_group(z[workspace], name=workspace)
            counts_path = f"matrices/{name}/counts"
            counts_t_path = f"matrices/{name}/countsT"
            matrix_group = as_zarr_group(
                matrix_root[f"matrices/{name}"],
                name=f"matrices/{name}",
            )
            self.rawData = ChunkedArray(
                as_zarr_array(matrix_root[counts_path], name=counts_path),
                nthreads=self.nthreads,
                resources=self.resources,
            )
            self.rawData._io = self.storageIo
            self.feats = MetaData(z[f"{workspace}/{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[f"{workspace}/{name}"], name=f"{workspace}/{name}")
        self.matrixGroup = matrix_group
        self.rawDataT: zarr.Array | None = None
        if "countsT" in matrix_group:
            try:
                counts_t = as_zarr_array(matrix_group["countsT"], name=counts_t_path)
            except TypeError:
                logger.warning(
                    f"({self.name}) Ignoring countsT at {counts_t_path}: "
                    "expected a Zarr array"
                )
            else:
                expected_shape = (self.rawData.shape[1], self.rawData.shape[0])
                if (
                    counts_t.attrs.get("complete") is True
                    and tuple(counts_t.shape) == expected_shape
                    and np.dtype(counts_t.dtype) == np.dtype(self.rawData.dtype)
                ):
                    self.rawDataT = counts_t
                else:
                    logger.warning(
                        f"({self.name}) Ignoring countsT at {counts_t_path}: "
                        "incomplete or mismatched with counts"
                    )
        self.attrs = self.z.attrs
        self.normMethod: NormMethod = norm_dummy
        self.sf: int | None = None
        self.scalar: np.ndarray | None = None
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None
        self._deferred_feature_props = False
        self._ini_feature_props()

    def _percent_features(self) -> PercentFeatures:
        raw = self.attrs.get("percentFeatures", {})
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data.

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix.
                      Defaults to the complete physical feature axis.
            **kwargs:

        Returns: A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = np.arange(self.feats.N, dtype=np.int64)
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)

    def to_raw_sparse(self, cell_key: str) -> csr_matrix:
        """

        Args:
            cell_key: A column from cell attribute table. This column must be a boolean
                      type. The data will be exported for only those that have a True value
                      in this column.

        Returns: A sparse matrix containing raw data.

        """
        sm = None
        selected = self.rawData[self.cells.active_index(cell_key), :]
        for values in selected.stream_blocks(
            nthreads=self.nthreads,
            msg=f"Converting {self.name} raw data to CSR",
        ):
            s = csr_matrix(values)
            if sm is None:
                sm = s
            else:
                sm = vstack([sm, s])
        return sm  # type: ignore

    def _ini_feature_props(self) -> None:
        """ """
        if "nCells" in self.feats.columns and "dropOuts" in self.feats.columns:
            return
        if _DEFER_FEATURE_PROPS.get():
            self._deferred_feature_props = True
            return
        ncells = compute_with_progress(
            (self.rawData > 0).sum(axis=0),
            f"({self.name}) Computing nCells and dropOuts",
            self.nthreads,
        )
        self._store_feature_props(ncells)

    def _store_feature_props(self, ncells: np.ndarray) -> None:
        self.feats.insert("nCells", ncells, overwrite=True)
        self.feats.insert(
            "dropOuts",
            abs(self.cells.N - self.feats.fetch_all("nCells")),
            overwrite=True,
        )
        self._deferred_feature_props = False

    def _stream_initialization_stats(
        self,
        *,
        compute_n_counts: bool,
        compute_n_features: bool,
        compute_n_cells: bool,
        percent_feature_indices: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        n_cells, n_features = self.rawData.shape
        sum_dtype = np.asarray(np.empty(0, dtype=self.rawData.dtype).sum()).dtype
        stats: dict[str, np.ndarray] = {}
        if compute_n_counts:
            stats["nCounts"] = np.empty(n_cells, dtype=sum_dtype)
        if compute_n_features:
            stats["nFeatures"] = np.empty(n_cells, dtype=np.int64)
        if compute_n_cells:
            stats["nCells"] = np.zeros(n_features, dtype=np.int64)
        for name in percent_feature_indices:
            stats[name] = np.empty(n_cells, dtype=sum_dtype)

        from ..storage.execution import (
            ExecutionReport,
            WorkShape,
            plan_operation,
            record_execution_report,
        )
        from ..storage.parallel import map_shards
        from ..utils.compute import pairwise_merge_tree

        ranges = self.rawData._ranges()
        largest_rows = max((end - start for start, end in ranges), default=1)
        matrix_elements = largest_rows * max(1, n_features)
        geometry = self.rawData._geometry()
        chunks_per_range = (
            1
            if geometry is None
            else max(
                1,
                -(-largest_rows // geometry.axisChunk(0))
                * -(-n_features // geometry.axisChunk(1)),
            )
        )
        decode_bytes = self.rawData._max_decode_bytes()
        unit_bytes = matrix_elements * max(1, int(self.rawData.dtype.itemsize))
        if compute_n_features or compute_n_cells:
            unit_bytes += matrix_elements
        if compute_n_counts:
            unit_bytes += largest_rows * max(1, int(sum_dtype.itemsize))
        if compute_n_features:
            unit_bytes += largest_rows * np.dtype(np.int64).itemsize
        if compute_n_cells:
            unit_bytes += n_features * np.dtype(np.int64).itemsize
        unit_bytes += (
            len(percent_feature_indices)
            * largest_rows
            * max(1, int(sum_dtype.itemsize))
        )
        plan = plan_operation(
            self.resources,
            WorkShape(
                nUnits=max(1, len(ranges)),
                unitBytes=max(1, unit_bytes),
                innerReadBytes=decode_bytes,
                chunksPerShard=chunks_per_range,
                ordered=False,
            ),
            policy=self.storageIo,
        )
        compute_slots = threading.Semaphore(plan.computeWorkers)

        def produce(
            index: int, start: int, end: int
        ) -> tuple[
            int,
            int,
            int,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            dict[str, np.ndarray],
            float,
            float,
        ]:
            fetch_started = time.perf_counter()
            raw = self.rawData._materialize_range(start, end)
            fetch_seconds = time.perf_counter() - fetch_started
            with compute_slots:
                compute_started = time.perf_counter()
                n_counts = raw.sum(axis=1) if compute_n_counts else None
                positive = None
                if compute_n_features or compute_n_cells:
                    positive = raw > 0
                n_features = (
                    positive.sum(axis=1)
                    if compute_n_features and positive is not None
                    else None
                )
                n_cells_partial = (
                    positive.sum(axis=0)
                    if compute_n_cells and positive is not None
                    else None
                )
                percents = {
                    name: raw[:, feat_idx].sum(axis=1)
                    for name, feat_idx in percent_feature_indices.items()
                }
                compute_seconds = time.perf_counter() - compute_started
            return (
                index,
                start,
                end,
                n_counts,
                n_features,
                n_cells_partial,
                percents,
                fetch_seconds,
                compute_seconds,
            )

        worker_count = min(plan.readWorkers, max(1, len(ranges)))
        results = map_shards(
            ranges,
            produce,
            workers=worker_count,
            within_block_threads=plan.threadsPerComputeWorker,
            io_concurrency=plan.ioConcurrency,
            msg=f"Computing {self.name} initialization statistics",
        )
        covered = 0
        fetch_seconds = 0.0
        compute_seconds = 0.0
        n_cell_partials: list[tuple[int, np.ndarray]] = []
        for (
            index,
            start,
            end,
            n_counts,
            n_features,
            n_cells_partial,
            percents,
            unit_fetch_seconds,
            unit_compute_seconds,
        ) in results:
            covered += end - start
            fetch_seconds += unit_fetch_seconds
            compute_seconds += unit_compute_seconds
            if compute_n_counts:
                assert n_counts is not None
                stats["nCounts"][start:end] = n_counts
            if compute_n_features:
                assert n_features is not None
                stats["nFeatures"][start:end] = n_features
            if compute_n_cells:
                assert n_cells_partial is not None
                n_cell_partials.append((index, np.asarray(n_cells_partial)))
            for name, values in percents.items():
                stats[name][start:end] = values
        record_execution_report(
            ExecutionReport(
                plan=plan,
                unitKind="initializationRowBand",
                actualReadWorkers=worker_count,
                actualComputeWorkers=min(plan.computeWorkers, worker_count),
                actualWriteWorkers=1,
                fetchSeconds=fetch_seconds,
                computeSeconds=compute_seconds,
                unitsCompleted=len(results),
                extra={
                    "effectiveChunkReadsInFlight": (worker_count * plan.ioConcurrency),
                    "fusedReadCompute": True,
                },
            )
        )
        if compute_n_cells and n_cell_partials:
            n_cell_partials.sort(key=lambda item: item[0])
            stats["nCells"] = pairwise_merge_tree(
                [item[1] for item in n_cell_partials],
                lambda left, right: left + right,
            )
        row_start = covered

        if row_start != n_cells:
            raise RuntimeError(
                f"({self.name}) Initialization stream produced {row_start} rows; "
                f"expected {n_cells}"
            )
        return stats

    def _plan_percent_feature(
        self,
        feat_pattern: str,
        name: str,
    ) -> np.ndarray | None:
        percent_features = self._percent_features()
        if name in percent_features:
            if percent_features[name] == feat_pattern:
                return None
            logger.debug(f"Pattern for percentage feature {name} updated")
        self.attrs["percentFeatures"] = {
            **percent_features,
            **{name: feat_pattern},
        }
        feat_idx = sorted(
            self.feats.get_index_by(self.feats.grep(feat_pattern), "names")
        )
        if len(feat_idx) == 0:
            logger.warning(
                f"No matches found for pattern {feat_pattern}."
                f" Will not add/update percentage feature"
            )
            return None
        return np.asarray(feat_idx, dtype=np.int64)

    def _write_percent_feature(
        self,
        name: str,
        total: np.ndarray,
        *,
        n_counts: np.ndarray | None = None,
    ) -> None:
        if total.sum() == 0:
            logger.warning(
                f"Percentage feature {name} not added because not detected in any cell"
            )
            return
        if n_counts is None:
            n_counts = self.cells.fetch_all(self.name + "_nCounts")
        self.cells.insert(
            name,
            100 * total / n_counts,
            overwrite=True,
        )

    def _compute_feature_percentage(
        self,
        cell_index: np.ndarray,
        feature_index: np.ndarray,
    ) -> np.ndarray:
        """Compute selected-feature count percentages in bounded row blocks."""
        values = np.empty(len(cell_index), dtype=np.float64)
        offset = 0
        selected = self.rawData[cell_index, :]
        for block in selected.stream_blocks(
            nthreads=self.nthreads,
            msg=f"({self.name}) Computing selected-feature percentages",
        ):
            counts = np.asarray(block)
            denominator = np.asarray(counts.sum(axis=1), dtype=np.float64)
            numerator = np.asarray(
                counts[:, feature_index].sum(axis=1),
                dtype=np.float64,
            )
            stop = offset + len(counts)
            values[offset:stop] = np.divide(
                100.0 * numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            )
            offset = stop
        if offset != len(values):
            raise RuntimeError(
                f"({self.name}) Percentage-feature stream produced {offset} rows; "
                f"expected {len(values)}"
            )
        return values

    def _get_cell_idx(self, cell_key: str) -> np.ndarray:
        """Validate and return the physical indices selected by ``cell_key``."""
        if cell_key not in self.cells.columns or self.cells.get_dtype(cell_key) != bool:  # noqa: E721
            raise ValueError(
                f"ERROR: Either {cell_key} does not exist or is not bool type"
            )
        return self.cells.active_index(cell_key)

    @staticmethod
    def _create_subset_hash(cell_idx: np.ndarray, feat_idx: np.ndarray) -> str:
        """Return a stable content digest for ordered cell and feature selections.

        The digest is persisted as a normalized-data cache key, so it must be
        deterministic across processes and Python runtimes.
        """
        cells = np.ascontiguousarray(np.asarray(cell_idx), dtype=np.int64)
        feats = np.ascontiguousarray(np.asarray(feat_idx), dtype=np.int64)
        # Prefix the cell count so the cell/feature boundary is encoded. Without
        # it, concatenation alone lets different splits (e.g. cells=[0,1],
        # feats=[2,3] versus cells=[0,1,2], feats=[3]) collide to one digest.
        boundary = np.array([cells.shape[0]], dtype=np.int64)
        return array_digest(np.concatenate([boundary, cells, feats]))

    def _write_normalized_payload(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        location: str,
        *,
        log_transform: bool,
        renormalize_subset: bool,
        mirror: zarr.Array | None = None,
    ) -> ChunkedArray:
        """Write one planned normalization artifact payload."""

        from ..storage.materialize import chunked_to_zarr

        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")
        if (
            np.any(cell_idx < 0)
            or np.any(cell_idx >= self.cells.N)
            or np.any(feat_idx < 0)
            or np.any(feat_idx >= self.feats.N)
        ):
            raise IndexError("cell_idx or feat_idx contains an out-of-range index")
        if location not in self.z:
            self.z.create_group(location)
        if location + "/data" in self.z:
            return ChunkedArray(
                as_zarr_array(self.z[location + "/data"], name=location + "/data"),
                nthreads=self.nthreads,
                resources=self.resources,
            )
        vals = self.normed(
            cell_idx,
            feat_idx,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
        )
        chunked_to_zarr(
            vals,
            self.z,
            location + "/data",
            self.nthreads,
            mirror=mirror,
            resources=self.resources,
            stats_group=as_zarr_group(self.z[location], name=location),
        )
        return ChunkedArray(
            as_zarr_array(self.z[location + "/data"], name=location + "/data"),
            nthreads=self.nthreads,
            resources=self.resources,
        )

    def iter_normed_feature_wise(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int | None,
        msg: str | None,
        as_dataframe: bool = True,
        **norm_params: Any,
    ) -> Generator[pd.DataFrame | tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate over explicitly selected normalized features in batches.

        Args:
            cell_idx: Ordered physical cell indices to include.
            feat_idx: Ordered physical feature indices to include.
            batch_size: Number of genes loaded at a time. When None, selected
                features are grouped into chunk-aligned blocks that fit the
                operation memory budget.
            msg: Message to be displayed in the progress bar
            as_dataframe: If true (default) then the yielded matrices are pandas dataframe
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            Generator yielding DataFrames or (matrix, feature index) tuples.
        """
        from ..storage.feature_stream import plan_feature_stream
        from ..utils.progress import iter_progress

        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")
        if msg is None:
            msg = ""
        data: ChunkedArray = self.normed(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            **norm_params,
        )
        backing = cast(zarr.Array, self.rawData._backing)
        raw_itemsize = max(1, int(np.dtype(backing.dtype).itemsize))
        out_itemsize = max(1, int(np.dtype(data.dtype).itemsize))
        n_cells = len(cell_idx)
        plan = plan_feature_stream(
            backing,
            featureAxis=1,
            cellAxis=0,
            featureIndices=feat_idx,
            cellIndices=cell_idx,
            resources=self.resources,
            blockBytes=lambda width: max(
                1,
                n_cells * width * (raw_itemsize + 2 * out_itemsize),
            ),
            requestedBatchSize=batch_size,
        )
        logger.debug(
            f"Will iterate over data of shape {data.shape} "
            f"in {len(plan.blocks)} feature blocks"
        )
        for block in iter_progress(plan.blocks, desc=msg, total=len(plan.blocks)):
            chunk = block.destinations
            if as_dataframe:
                yield pd.DataFrame(
                    controlled_compute(data[:, chunk], self.nthreads),
                    columns=block.indices,
                )
            else:
                yield (
                    controlled_compute(data[:, chunk], self.nthreads).T,
                    block.indices,
                )

    def _prepare_aggregated_ordering(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        cell_ordering: np.ndarray,
        *,
        min_exp: float,
        window_size: int,
        chunk_size: int,
        smoothen: bool,
        z_scale: bool,
        norm_params: dict[str, Any],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        list[str],
        dict[str, Any],
    ]:
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        cell_ordering = np.asarray(cell_ordering, dtype=float)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")
        if len(cell_idx) == 0 or len(feat_idx) == 0:
            raise ValueError("Aggregation requires non-empty cell and feature indices")
        if (
            np.any(cell_idx < 0)
            or np.any(cell_idx >= self.cells.N)
            or np.unique(cell_idx).size != len(cell_idx)
            or np.any(feat_idx < 0)
            or np.any(feat_idx >= self.feats.N)
            or np.unique(feat_idx).size != len(feat_idx)
        ):
            raise ValueError("Aggregation indices are invalid")
        n_cells = cell_ordering.shape[0]
        if cell_ordering.ndim != 1 or n_cells == 0:
            raise ValueError("Cell ordering must be a non-empty one-dimensional array")
        if not np.isfinite(cell_ordering).all():
            raise ValueError("Cell ordering must contain only finite values")
        if n_cells != len(cell_idx):
            raise ValueError("Cell ordering must align with cell_idx")
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("window_size must be an integer")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        effective_window = min(window_size, n_cells)
        effective_bins = min(chunk_size, n_cells)
        if effective_window != window_size:
            logger.warning(
                f"Reducing window_size from {window_size} to {effective_window} "
                "for the selected cell count"
            )
        if effective_bins != chunk_size:
            logger.warning(
                f"Reducing chunk_size from {chunk_size} to {effective_bins} "
                "for the selected cell count"
            )
        hashes = [array_digest(x) for x in (cell_idx, feat_idx, cell_ordering)]
        params = {
            "min_exp": min_exp,
            "window_size": window_size,
            "effective_window": effective_window,
            "chunk_size": chunk_size,
            "effective_bins": effective_bins,
            "smoothen": smoothen,
            "z_scale": z_scale,
            "norm_params": norm_params,
        }
        return (
            cell_ordering,
            cell_idx,
            feat_idx,
            effective_window,
            effective_bins,
            hashes,
            params,
        )

    def _write_aggregated_ordering_group(
        self,
        group: zarr.Group,
        *,
        cell_idx: np.ndarray,
        cell_ordering: np.ndarray,
        feat_idx: np.ndarray,
        min_exp: float,
        effective_window: int,
        effective_bins: int,
        smoothen: bool,
        z_scale: bool,
        batch_size: int | None,
        norm_params: dict[str, Any],
    ) -> tuple[ChunkedArray, np.ndarray, np.ndarray]:
        from ..storage.arrays import create_numeric_array, create_zarr_dataset
        from ..storage.layout import row_sharded_array_spec
        from ..storage.profiles import resolve_storage_profile
        from ..trajectory.feature_dynamics import aggregate_feature_profiles

        aggregated_shape = (int(feat_idx.shape[0]), int(effective_bins))
        data_array = create_numeric_array(
            group,
            "data",
            row_sharded_array_spec(
                aggregated_shape,
                "float64",
                profile=resolve_storage_profile(group.store),
                band_rows=max(1, aggregated_shape[0]),
            ),
        )
        ordering_idx = np.argsort(cell_ordering, kind="stable")
        stored_feat_idx: list[int] = []
        valid_feat_flags: list[bool] = []
        offset = 0
        for item in self.iter_normed_feature_wise(
            cell_idx,
            feat_idx,
            batch_size,
            "Binning over cell-ordering",
            True,
            **norm_params,
        ):
            frame = cast(pd.DataFrame, item)
            stored_feat_idx.extend(list(frame.columns))
            aggregated, valid_features = aggregate_feature_profiles(
                frame.to_numpy(dtype=float),
                ordering_idx,
                np.asarray(frame.columns),
                min_expression=min_exp,
                window_size=effective_window,
                n_bins=effective_bins,
                smooth=smoothen,
                z_scale=z_scale,
            )
            valid_feat_flags.extend(valid_features.tolist())
            data_array[offset : offset + aggregated.shape[0]] = aggregated
            offset += aggregated.shape[0]

        feature_indices = np.asarray(stored_feat_idx, dtype=np.uint64)
        valid = np.asarray(valid_feat_flags, dtype=bool)
        feature_array = create_zarr_dataset(
            group,
            "feature_indices",
            (min(max(len(feature_indices), 1), 100_000),),
            "uint64",
            (len(feature_indices),),
        )
        feature_array[:] = feature_indices
        valid_array = create_zarr_dataset(
            group,
            "valid_features",
            (min(max(len(valid), 1), 100_000),),
            "bool",
            (len(valid),),
        )
        valid_array[:] = valid
        return (
            ChunkedArray(
                data_array,
                nthreads=self.nthreads,
                resources=self.resources,
            ),
            feature_indices,
            valid,
        )

    def mean_features(
        self,
        feature_names: Sequence[str],
        cell_key: str = "I",
        *,
        missing: Literal["error", "skip"] = "error",
    ) -> np.ndarray:
        """Per-cell mean normalized expression over named features.

        Returns one value per active cell under ``cell_key``. Does not write
        cell metadata. Distinct from ``score_features``, which subtracts a
        control-gene background.
        """
        from .rna import RNAassay

        if missing not in ("error", "skip"):
            raise ValueError("missing must be 'error' or 'skip'")
        if not feature_names:
            raise ValueError("feature_names must be non-empty")

        requested = [str(name) for name in feature_names]
        if len(set(name.upper() for name in requested)) != len(requested):
            raise ValueError("feature_names contains duplicate names")

        name_to_indices: dict[str, list[int]] = {}
        for index, name in enumerate(self.feats.fetch_all("names")):
            key = str(name).upper()
            name_to_indices.setdefault(key, []).append(index)

        feature_idx: list[int] = []
        missing_names: list[str] = []
        for name in requested:
            matches = name_to_indices.get(name.upper(), [])
            if not matches:
                missing_names.append(name)
                continue
            if len(matches) > 1:
                raise ValueError(f"Feature name {name!r} matches multiple features")
            feature_idx.append(matches[0])

        if missing_names:
            if missing == "error":
                raise ValueError("Features not found: " + ", ".join(missing_names))
            if not feature_idx:
                raise ValueError("No requested features were found")

        cell_idx = self._get_cell_idx(cell_key)
        feat_idx = np.asarray(feature_idx, dtype=int)
        if isinstance(self, RNAassay) and self.normMethod is norm_lib_size:
            means = self._mean_normed_feature_groups(
                cell_idx,
                {"target": feat_idx},
            )
            return np.asarray(means["target"])
        return np.asarray(
            self.normed(cell_idx=cell_idx, feat_idx=np.sort(feat_idx))
            .mean(axis=1)
            .compute()
        )

    def score_features(
        self,
        feature_names: list[str],
        cell_key: str,
        ctrl_size: int,
        n_bins: int,
        rand_seed: int,
    ) -> np.ndarray:
        """Calculates the scores (mean values) of selection of features over a
        randomly sampled selected feature set in given cells (as marked by
        cell_key)

        Args:
            feature_names: Names (as in 'names' column of the feature attribute table) of features to
                           be used for scoring
            cell_key: Name of the key (column) from cell attribute table.
            ctrl_size: Number of reference features to be sampled from each bin.
            n_bins: Number of bins for sampling.
            rand_seed: The seed to use for the random number generation.

        Returns: Numpy array of the calculated scores
        """

        from .rna import RNAassay

        feature_idx = self.feats.get_index_by(feature_names, "names", None)
        if len(feature_idx) == 0:
            raise ValueError(
                f"ERROR: No feature ids found for any of the provided {len(feature_names)} features"
            )
        cell_idx = self._get_cell_idx(cell_key)
        if isinstance(self, RNAassay) and self.normMethod is norm_lib_size:
            summary = self._compute_feature_summary(
                cell_idx,
                np.arange(self.feats.N, dtype=np.int64),
            )
            totals = np.asarray(summary["normed_tot"], dtype=np.float64)
            obs_avg = (
                totals / len(cell_idx)
                if len(cell_idx) > 0
                else np.zeros(self.feats.N, dtype=np.float64)
            )
        elif len(cell_idx) > 0:
            obs_avg = np.asarray(
                self.normed(
                    cell_idx=cell_idx,
                    feat_idx=np.arange(self.feats.N, dtype=np.int64),
                )
                .mean(axis=0)
                .compute(),
                dtype=np.float64,
            )
        else:
            obs_avg = np.zeros(self.feats.N, dtype=np.float64)
        return self._score_feature_indices(
            np.asarray(feature_idx, dtype=np.int64),
            cell_idx,
            obs_avg,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
        )

    def _compute_feature_summary(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute full-axis sufficient statistics for supported assay types."""
        raise TypeError(
            "Feature summaries are supported only for RNAassay and ATACassay"
        )

    def _score_feature_indices(
        self,
        feature_idx: np.ndarray,
        cell_idx: np.ndarray,
        feature_avg: np.ndarray,
        *,
        ctrl_size: int,
        n_bins: int,
        rand_seed: int,
    ) -> np.ndarray:
        """Score feature indexes against controls using supplied feature means."""
        from ..features.scoring import binned_sampling
        from .rna import RNAassay

        feature_idx = np.asarray(feature_idx, dtype=np.int64)
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feature_avg = np.asarray(feature_avg, dtype=np.float64)
        if feature_idx.ndim != 1 or len(feature_idx) == 0:
            raise ValueError("feature_idx must be a non-empty one-dimensional array")
        if feature_avg.shape != (self.feats.N,):
            raise ValueError(
                f"feature_avg must have shape ({self.feats.N},), got "
                f"{feature_avg.shape}"
            )
        control_idx = np.asarray(
            binned_sampling(
                pd.Series(feature_avg),
                feature_idx.tolist(),
                ctrl_size,
                n_bins,
                rand_seed,
            ),
            dtype=np.int64,
        )

        if isinstance(self, RNAassay) and self.normMethod is norm_lib_size:
            means = self._mean_normed_feature_groups(
                cell_idx,
                {
                    "target": feature_idx,
                    "control": control_idx,
                },
            )
            return np.asarray(means["target"] - means["control"])

        def calc_mean(index: np.ndarray) -> np.ndarray:
            return np.asarray(
                self.normed(cell_idx=cell_idx, feat_idx=np.sort(index))
                .mean(axis=1)
                .compute()
            )

        return np.asarray(calc_mean(feature_idx) - calc_mean(control_idx))

    def __repr__(self) -> str:
        assay_name = str(self.__class__).split(".")[-1][:-2]
        return f"{assay_name} {self.name} with {self.feats.N} features"
