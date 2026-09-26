from collections.abc import Generator, Iterator, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, vstack

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.artifacts import provenance_hash
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import array_digest, regex_match_mask
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from .normalization import (
    NormalizedValueSource,
    NormMethod,
    iter_feature_group_means,
    norm_dummy,
    norm_lib_size,
    normalizer_count_arithmetic,
)
from ..utils.arrays import has_duplicates

type PercentFeatures = dict[str, str]


def raw_csr(
    assay: "Assay",
    cell_idx: np.ndarray,
    feat_idx: np.ndarray | None = None,
) -> csr_matrix:
    """Return the raw counts of selected cells and features as one CSR matrix.

    Rows are converted in bounded blocks and stacked once, because stacking
    per block copies the growing matrix every time. An empty cell selection
    returns a matrix with zero rows.
    """
    counts = assay.rawData if feat_idx is None else assay.rawData[:, feat_idx]
    selected = counts[cell_idx, :]
    blocks = [
        csr_matrix(values)
        for values in selected.stream_blocks(
            nthreads=assay.nthreads,
            msg=f"Converting {assay.name} raw data to CSR",
        )
    ]
    if not blocks:
        return csr_matrix(selected.shape, dtype=assay.rawData.dtype)
    return cast(csr_matrix, vstack(blocks, format="csr"))


def _stream_byte_count(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


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
        from ..storage.counts_t_contract import validate_count_matrix

        _, self.rawDataT = validate_count_matrix(
            matrix_group,
            require_transpose=self.requiresCountsT or "countsT" in matrix_group,
        )
        self.attrs = self.z.attrs
        self.normMethod: NormMethod = norm_dummy
        self.sf: int | None = None
        self.scalar: np.ndarray | None = None
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None

    def _percent_features(self) -> PercentFeatures:
        raw = self.attrs.get("percentFeatures", {})
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def _cell_count_totals(self, cell_idx: np.ndarray) -> np.ndarray:
        """Read the prepared cell totals."""
        if len(cell_idx) == 0:
            return np.empty(0, dtype=np.float64)
        column = self.name + "_nCounts"
        totals = self.cells.fetch_all(column)[cell_idx]
        return np.asarray(totals, dtype=np.float64)

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
        from ..storage.identity import read_dataset_fingerprint

        read_dataset_fingerprint(self.z)
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

        Returns: A sparse matrix containing raw data. An empty cell selection
            returns a matrix with zero rows and one column per feature.

        """
        return raw_csr(self, self.cells.active_index(cell_key))

    requiresCountsT = False

    def prepare(self, percent_patterns: dict[str, str | None]) -> None:
        from ..storage.identity import (
            REBUILD_REQUIRED,
            clear_column,
            fresh_group,
            load_count_summaries,
            publish_preparation,
            validate_preparation,
        )

        self.z = fresh_group(self.z)
        self.attrs = self.z.attrs
        state = self.attrs.get("prepared")
        cells = self.cells.locations["primary"]
        if state is True:
            validate_preparation(
                self.z,
                cells,
                self.matrixGroup,
                require_transpose=self.requiresCountsT,
            )
            for name, pattern in percent_patterns.items():
                if pattern and self._plan_percent_feature(pattern, name) is not None:
                    raise ValueError(
                        f"Percentage {name!r} was not computed when assay "
                        f"{self.name!r} was first prepared, and a prepared assay "
                        "cannot add percentage columns. To use pattern "
                        f"{pattern!r}, import the data into a fresh store and "
                        "pass the pattern when that store is first opened, or "
                        "compute a separate quality-metric artifact with "
                        "run_feature_percentage and an explicit feature selection."
                    )
            return
        if state is not False or self.z.read_only:
            raise ValueError(f"Assay {self.name!r} is not prepared. {REBUILD_REQUIRED}")

        # First preparation derives every percentage from the counts, so any
        # imported column with a percentage name is replaced, never trusted.
        for name in sorted(set(percent_patterns) | set(self._percent_features())):
            if name in cells:
                logger.warning(
                    f"Discarding existing cell column {name!r}: the first "
                    f"preparation of assay {self.name!r} derives percentage "
                    "columns from its counts with the configured patterns."
                )
            clear_column(cells, name)
        self.attrs["percentFeatures"] = {}
        planned = {
            name: result
            for name, pattern in percent_patterns.items()
            if pattern
            and (result := self._plan_percent_feature(pattern, name)) is not None
        }
        n_counts, n_features, n_cells = load_count_summaries(
            self.matrixGroup, as_zarr_array(self.matrixGroup["counts"], name="counts")
        )
        totals = self._feature_totals(
            {name: result[0] for name, result in planned.items()}
        )
        self.cells.insert(f"{self.name}_nCounts", n_counts, overwrite=True)
        self.cells.insert(
            f"{self.name}_nFeatures", n_features.astype(np.float64), overwrite=True
        )
        self.feats.insert("nCells", n_cells, overwrite=True)
        self.feats.insert("dropOuts", self.cells.N - n_cells, overwrite=True)
        for name, (_, fingerprint) in planned.items():
            pattern = percent_patterns[name]
            assert pattern is not None
            self._write_percent_feature(
                name,
                totals[name],
                feat_pattern=pattern,
                feature_fingerprint=fingerprint,
                n_counts=n_counts,
            )
        publish_preparation(
            self.z,
            cells,
            self.matrixGroup,
            require_transpose=self.requiresCountsT,
        )
        self.z = fresh_group(self.z)
        self.attrs = self.z.attrs

    def _feature_totals(self, indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Sum each feature set per cell, reading only the selected features.

        Sums saved while ``countsT`` was written are reused when they cover
        exactly the same features of the same counts.
        """
        from ..storage.execution import WorkShape, plan_operation
        from ..storage.identity import load_feature_sums
        from ..storage.parallel import stream_shards

        counts = as_zarr_array(self.matrixGroup["counts"], name="counts")
        saved = {
            name: load_feature_sums(self.matrixGroup, counts, np.unique(values))
            for name, values in indices.items()
        }
        found = {name: sums for name, sums in saved.items() if sums is not None}
        indices = {
            name: values for name, values in indices.items() if name not in found
        }
        if not indices:
            return found
        wanted = np.unique(np.concatenate(list(indices.values())))
        positions = {
            name: np.searchsorted(wanted, values) for name, values in indices.items()
        }
        totals = {name: np.zeros(self.cells.N, dtype=np.float64) for name in indices}
        if self.rawDataT is None:
            offset = 0
            for block in self.rawData[:, wanted].stream_blocks(nthreads=self.nthreads):
                values = np.asarray(block, dtype=np.float64)
                for name, position in positions.items():
                    totals[name][offset : offset + len(values)] = values[
                        :, position
                    ].sum(axis=1)
                offset += len(values)
            return {**found, **totals}
        source = self.rawDataT
        cell_chunk = max(1, int(source.chunks[1]))
        column_bytes = 2 * len(wanted) * np.dtype(np.float64).itemsize
        target = min(64 * 1024**2, self.resources.memoryBytes // 4)
        band = max(
            1,
            min(
                self.cells.N,
                cell_chunk * max(1, target // (cell_chunk * column_bytes)),
            ),
        )
        bands = [
            (start, min(start + band, self.cells.N))
            for start in range(0, self.cells.N, band)
        ]
        operation = plan_operation(
            self.resources,
            WorkShape(nUnits=len(bands), unitBytes=band * column_bytes),
            policy=self.storageIo,
        )

        def add(bounds: tuple[int, int]) -> None:
            start, stop = bounds
            values = np.asarray(
                source.get_orthogonal_selection((wanted, slice(start, stop))),
                dtype=np.float64,
            )
            for name, position in positions.items():
                totals[name][start:stop] = values[position].sum(axis=0)

        for _ in stream_shards(
            bands,
            add,
            workers=operation.computeWorkers,
            io_concurrency=operation.ioConcurrency,
        ):
            pass
        return {**found, **totals}

    def _plan_percent_feature(
        self,
        feat_pattern: str,
        name: str,
    ) -> tuple[np.ndarray, str] | None:
        feat_idx = np.flatnonzero(
            regex_match_mask(self.feats.fetch_all("names"), feat_pattern)
        )
        fingerprint = provenance_hash(
            {
                "pattern": feat_pattern,
                "feature_indices": feat_idx.tolist(),
                "feature_ids": self.feats.fetch_all("ids")[feat_idx].tolist(),
            }
        )
        percent_features = self._percent_features()
        has_column = name in self.cells.columns
        if has_column:
            if (
                percent_features.get(name) == feat_pattern
                and self.cells._get_array(name).attrs.get(
                    "feature_selection_fingerprint"
                )
                == fingerprint
            ):
                return None
            raise ValueError(
                f"Cannot apply pattern {feat_pattern!r} to existing {name}: "
                "its recorded pattern or matched-feature provenance differs or "
                "is missing. Omit the pattern to preserve the stored values. "
                "Use run_feature_percentage with an explicit feature selection "
                "to compute a separate quality-metric artifact."
            )
        if len(feat_idx) == 0:
            logger.warning(
                f"No matches found for pattern {feat_pattern}. "
                f"Percentage feature {name} is unavailable"
            )
            return None
        return feat_idx, fingerprint

    def _write_percent_feature(
        self,
        name: str,
        total: np.ndarray,
        *,
        feat_pattern: str,
        feature_fingerprint: str,
        n_counts: np.ndarray | None = None,
    ) -> None:
        if n_counts is None:
            n_counts = self.cells.fetch_all(self.name + "_nCounts")
        self.cells.insert(
            name,
            np.divide(
                100 * total,
                n_counts,
                out=np.full(total.shape, np.nan, dtype=np.float64),
                where=n_counts != 0,
            ),
            overwrite=False,
        )
        self.cells._get_array(name).attrs["feature_selection_fingerprint"] = (
            feature_fingerprint
        )
        self.attrs["percentFeatures"] = {
            **self._percent_features(),
            name: feat_pattern,
        }

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

    def _count_arithmetic(
        self,
        values: NormalizedValueSource,
        *,
        log_transform: bool = False,
        renormalize_subset: bool = False,
    ) -> Literal["float64"] | None:
        """Return the count-arithmetic marker of an artifact of these values.

        ``values`` names what the artifact reads: ``normed`` itself, the
        ``run_normalization`` payload, ``iter_normed_feature_wise`` batches,
        or feature scores. This assay computes all of them with ``normed``,
        which ignores the normalization flags.
        """
        return normalizer_count_arithmetic(self, self.normMethod)

    def _iter_feature_group_means(
        self,
        cell_idx: np.ndarray,
        feature_groups: Sequence[np.ndarray],
        *,
        block_rows: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield per-cell group means, as ``iter_feature_group_means`` does.

        ``block_rows`` is the preferred row band of assays that read counts
        directly. Normalized blocks here keep their budgeted size, because
        their values do not depend on it.
        """
        yield from iter_feature_group_means(self, cell_idx, feature_groups)

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
        scratch_itemsize: int = 0,
        resident_bytes: int = 0,
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
            scratch_itemsize: Bytes of working memory the caller needs per
                yielded value while it processes one batch. Batches are sized
                so that this scratch also fits the memory budget.
            resident_bytes: Bytes the caller keeps allocated for the whole
                iteration, such as an output buffer.
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
        scratch_itemsize = _stream_byte_count(scratch_itemsize, "scratch_itemsize")
        resident_bytes = _stream_byte_count(resident_bytes, "resident_bytes")
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
                n_cells * width * (raw_itemsize + 2 * out_itemsize + scratch_itemsize),
            ),
            residentBytes=resident_bytes,
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
            or has_duplicates(cell_idx)
            or np.any(feat_idx < 0)
            or np.any(feat_idx >= self.feats.N)
            or has_duplicates(feat_idx)
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

    def _aggregate_ordering_profiles(
        self,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate normalized features along a cell ordering in memory.

        Returns the features-by-bins profiles, the streamed feature indices,
        and the mask of features that pass the expression and variability
        filter. Nothing is written, so a caller can validate the result before
        it starts an artifact.
        """
        from ..trajectory.feature_dynamics import (
            AGGREGATION_SCRATCH_ITEMSIZE,
            aggregate_feature_profiles,
        )

        n_features = int(feat_idx.shape[0])
        data = np.empty((n_features, int(effective_bins)), dtype=np.float64)
        feature_indices = np.empty(n_features, dtype=np.uint64)
        valid = np.empty(n_features, dtype=bool)
        ordering_idx = np.argsort(cell_ordering, kind="stable")
        resident_bytes = (
            data.nbytes + feature_indices.nbytes + valid.nbytes + ordering_idx.nbytes
        )
        offset = 0
        for item in self.iter_normed_feature_wise(
            cell_idx,
            feat_idx,
            batch_size,
            "Binning over cell-ordering",
            False,
            scratch_itemsize=AGGREGATION_SCRATCH_ITEMSIZE,
            resident_bytes=resident_bytes,
            **norm_params,
        ):
            values, labels = cast(tuple[np.ndarray, np.ndarray], item)
            del item
            aggregated, batch_valid = aggregate_feature_profiles(
                values.T,
                ordering_idx,
                labels,
                min_expression=min_exp,
                window_size=effective_window,
                n_bins=effective_bins,
                smooth=smoothen,
                z_scale=z_scale,
            )
            del values
            stop = offset + aggregated.shape[0]
            data[offset:stop] = aggregated
            feature_indices[offset:stop] = labels
            valid[offset:stop] = batch_valid
            offset = stop
        if offset != n_features:
            raise ValueError("Normalized features do not cover the selected features")
        return data, feature_indices, valid

    def _write_aggregated_ordering_group(
        self,
        group: zarr.Group,
        *,
        data: np.ndarray,
        feature_indices: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Write aggregated profiles into a started artifact group."""
        from ..storage.arrays import create_numeric_array, create_zarr_dataset
        from ..storage.layout import row_sharded_array_spec
        from ..storage.profiles import resolve_storage_profile

        aggregated_shape = (int(data.shape[0]), int(data.shape[1]))
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
        data_array[:] = data
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
        *,
        log_transform: bool = False,
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
                log_transform=log_transform,
            )
            totals = np.asarray(summary["normed_tot"], dtype=np.float64)
            obs_avg = (
                totals / len(cell_idx)
                if len(cell_idx) > 0
                else np.zeros(self.feats.N, dtype=np.float64)
            )
        elif len(cell_idx) > 0:
            values = self.normed(
                cell_idx=cell_idx,
                feat_idx=np.arange(self.feats.N, dtype=np.int64),
            )
            if log_transform:
                values = cast(ChunkedArray, np.log1p(values))
            obs_avg = np.asarray(values.mean(axis=0).compute(), dtype=np.float64)
        else:
            obs_avg = np.zeros(self.feats.N, dtype=np.float64)
        return self._score_feature_indices(
            np.asarray(feature_idx, dtype=np.int64),
            cell_idx,
            obs_avg,
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
            log_transform=log_transform,
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
        log_transform: bool = False,
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

        if len(control_idx) == 0:
            raise ValueError(
                "No control features were sampled. Reduce n_bins or increase ctrl_size."
            )
        if isinstance(self, RNAassay) and self.normMethod is norm_lib_size:
            means = self._mean_normed_feature_groups(
                cell_idx,
                {
                    "target": feature_idx,
                    "control": control_idx,
                },
                log_transform=log_transform,
            )
            return np.asarray(means["target"] - means["control"])

        def calc_mean(index: np.ndarray) -> np.ndarray:
            values = self.normed(cell_idx=cell_idx, feat_idx=np.sort(index))
            if log_transform:
                values = cast(ChunkedArray, np.log1p(values))
            return np.asarray(values.mean(axis=1).compute())

        return np.asarray(calc_mean(feature_idx) - calc_mean(control_idx))

    def __repr__(self) -> str:
        assay_name = str(self.__class__).split(".")[-1][:-2]
        return f"{assay_name} {self.name} with {self.feats.N} features"
