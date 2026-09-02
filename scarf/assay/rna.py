from collections.abc import Generator, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from numba import njit

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.budget import admit_stream
from ..storage.feature_stream import FeatureStreamPlan, plan_feature_stream
from ..storage.geometry import array_geometry
from ..storage.partition import IndexBlock, row_band
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.compute import compute_with_progress
from ..utils.logging import logger
from .base import Assay
from .normalization import (
    lib_size_feature_stream_eligible,
    norm_lib_size,
    norm_lib_size_log,
)


def _read_facade_block(
    zarr_arr: zarr.Array,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    from . import _read_block

    return _read_block(zarr_arr, row_idx, col_idx)


@njit(cache=True, nogil=True)
def _hvg_stats_gene_major_kernel(
    values: np.ndarray,
    inv: np.ndarray,
    sf: float,
    dest: np.ndarray,
    selected: np.ndarray,
    out_nz: np.ndarray,
    out_s1: np.ndarray,
    out_s2: np.ndarray,
) -> None:
    """Accumulate lib-size HVG stats over selected cells in a raw block."""
    n_genes = values.shape[0]
    n_selected = selected.shape[0]
    for g in range(n_genes):
        target = dest[g]
        if target < 0:
            continue
        c_nz = 0.0
        c_s1 = 0.0
        c_s2 = 0.0
        for i in range(n_selected):
            cell = selected[i]
            value = sf * np.float64(values[g, cell]) * inv[i]
            if value > 0.0:
                c_nz += 1.0
            c_s1 += value
            c_s2 += value * value
        out_nz[target] += c_nz
        out_s1[target] += c_s1
        out_s2[target] += c_s2


def _hvg_stats_gene_major(
    values: np.ndarray,
    inv: np.ndarray,
    sf: float,
    dest: np.ndarray,
    out_nz: np.ndarray,
    out_s1: np.ndarray,
    out_s2: np.ndarray,
    selected: np.ndarray | None = None,
) -> None:
    """Accumulate lib-size HVG stats for a gene-major count block."""
    if selected is None:
        selected_cells = np.arange(int(values.shape[1]), dtype=np.int64)
    else:
        selected_cells = np.asarray(selected, dtype=np.int64)
    _hvg_stats_gene_major_kernel(
        values,
        inv,
        sf,
        dest,
        selected_cells,
        out_nz,
        out_s1,
        out_s2,
    )


def _as_feature_indexes(
    values: Sequence[int],
    *,
    n_features: int,
    name: str,
    require_unique: bool,
) -> np.ndarray:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of integer feature indexes")
    indexes = np.asarray(values)
    if indexes.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if indexes.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(indexes.dtype, np.integer):
        raise TypeError(f"{name} must contain only integer feature indexes")
    indexes = indexes.astype(np.int64, copy=False)
    if np.any(indexes < 0) or np.any(indexes >= n_features):
        raise IndexError(f"{name} contains an out-of-range feature index")
    if require_unique and np.unique(indexes).size != indexes.size:
        raise ValueError(f"{name} contains duplicate feature indexes")
    return indexes


def _corrected_variance_column(
    n_bins: int,
    lowess_frac: float,
    bin_strategy: Literal["fixed", "adaptive"],
) -> str:
    if bin_strategy not in ("fixed", "adaptive"):
        raise ValueError("bin_strategy must be either 'fixed' or 'adaptive'")
    if bin_strategy == "adaptive":
        if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
            n_bins,
            (int, np.integer),
        ):
            raise TypeError("n_bins must be an integer")
        if n_bins < 1:
            raise ValueError("n_bins must be greater than 0")
        if isinstance(lowess_frac, (bool, np.bool_)) or not isinstance(
            lowess_frac,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("lowess_frac must be numeric")
        if not np.isfinite(lowess_frac) or not 0 <= lowess_frac <= 1:
            raise ValueError("lowess_frac must be between 0 and 1")
        return f"c_var__adaptive__{n_bins}__{lowess_frac}"
    if not 0 <= lowess_frac <= 1:
        raise ValueError("lowess_frac must be between 0 and 1")
    return f"c_var__{n_bins}__{lowess_frac}"


class RNAassay(Assay):
    """This subclass of Assay is designed for feature selection and
    normalization of scRNA-Seq data.

    Args:
        z (zarr.Group): Zarr hierarchy where raw data is located
        name (str): A label/name for assay.
        cell_data: Metadata class object for the cell attributes.
        **kwargs: kwargs to be passed to the Assay class

    Attributes:
        normMethod: A pointer to the function to be used for normalization of the raw data
        sf: scaling factor for doing library-size normalization
        scalar: This is used to cache the library size of the cells.
                It is set to None until normed method is called.
    """

    _feature_summary_operation = "summarize_rna_features"

    def __init__(
        self,
        z: zarr.Group,
        name: str,
        cell_data: MetaData,
        *,
        workspace: str | None = None,
        nthreads: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            z=z,
            workspace=workspace,
            name=name,
            cell_data=cell_data,
            nthreads=nthreads,
            **kwargs,
        )
        self.normMethod = norm_lib_size
        if "size_factor" in self.attrs:
            self.sf = int(cast(int, self.attrs["size_factor"]))
        else:
            self.sf = 1000
            self.attrs["size_factor"] = self.sf
        self.scalar: np.ndarray | None = None
        self._require_counts_t()

    def _require_counts_t(self) -> None:
        """RNA assays require complete sharded ``countsT`` on Zarr v3."""
        from ..storage.count_matrix import require_count_matrix_layout

        # Stub construction (tests that monkeypatch Assay.__init__) skips load.
        if not hasattr(self, "rawDataT"):
            return
        if self.rawDataT is None:
            raise ValueError(
                f"RNA assay {self.name!r} requires a complete sharded "
                "countsT matrix. Rebuild with ingest/subset/merge on Zarr v3, "
                "or run repack_zarr / write_counts_t."
            )
        counts_t = self.rawDataT
        zarr_format = int(getattr(counts_t.metadata, "zarr_format", 3) or 3)
        if zarr_format < 3:
            raise ValueError(
                f"RNA assay {self.name!r} requires Zarr v3 for sharded "
                "countsT. Repack the store to Zarr v3."
            )
        counts = as_zarr_array(self.rawData._backing, name="counts")
        matrix_group = as_zarr_group(self.matrixGroup, name="matrix")
        require_count_matrix_layout(matrix_group, counts, counts_t)

    def iter_normed_feature_wise(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int | None,
        msg: str | None,
        as_dataframe: bool = True,
        **norm_params: Any,
    ) -> Generator[pd.DataFrame | tuple[np.ndarray, np.ndarray], None, None]:
        renormalize_subset = bool(norm_params.get("renormalize_subset", False))
        if not lib_size_feature_stream_eligible(
            self, renormalize_subset=renormalize_subset
        ):
            yield from Assay.iter_normed_feature_wise(
                self,
                cell_idx,
                feat_idx,
                batch_size,
                msg,
                as_dataframe=as_dataframe,
                **norm_params,
            )
            return

        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")

        if msg is None:
            msg = ""

        sf = self.sf
        if sf is None:
            raise ValueError("RNA library-size normalization requires a size factor")
        scalar = self.cells.fetch_all(self.name + "_nCounts")[cell_idx]
        log_transform = bool(norm_params.get("log_transform", False))
        counts_t = self.rawDataT
        if counts_t is None:
            raise ValueError(
                f"RNA assay {self.name!r} requires sharded countsT "
                "for feature-wise streaming"
            )
        scalar_values = np.asarray(scalar, dtype=np.float32)
        scalar_values[scalar_values == 0] = 1
        n_feats = int(counts_t.shape[0])
        dest_of = np.full(n_feats, -1, dtype=np.int64)
        dest_of[feat_idx] = np.arange(len(feat_idx), dtype=np.int64)
        feat_labels = np.asarray(feat_idx)
        from ..storage.feature_stream import (
            map_feature_read_groups,
            selected_feature_values,
        )

        extra_itemsize = int(np.dtype(np.float32).itemsize) + int(
            np.dtype(np.float64).itemsize
        )
        loaded_groups = map_feature_read_groups(
            counts_t,
            lambda loaded: loaded,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=self.resources,
            progress=msg or None,
            io=getattr(self, "storageIo", None),
            extraItemsize=extra_itemsize,
            orderedCompute=True,
        )

        def selected_values(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
            return selected_feature_values(values, keep)

        resolved_batch = None if batch_size is None else max(1, int(batch_size))
        pending_cols: np.ndarray | None = None
        pending_labels: np.ndarray | None = None

        def emit(
            cols: np.ndarray, labels: np.ndarray
        ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
            if as_dataframe:
                return pd.DataFrame(np.asarray(cols, dtype=np.float64), columns=labels)
            return np.asarray(cols.T, dtype=np.float64), labels

        for group in loaded_groups:
            local_dest = dest_of[group.featStart : group.featEnd]
            keep = local_dest >= 0
            if not np.any(keep):
                continue
            raw = selected_values(group.values, keep)
            destinations = local_dest[keep]
            # cells x features for downstream consumers
            mat = raw.T.astype(np.float32, copy=False)
            mat *= float(sf)
            mat /= scalar_values[:, None]
            if log_transform:
                np.log1p(mat, out=mat)
            labels = feat_labels[destinations]
            cols = np.asarray(mat, dtype=np.float64)
            del mat, raw
            if resolved_batch is None:
                yield emit(cols, labels)
                continue
            if pending_cols is not None:
                assert pending_labels is not None
                need = resolved_batch - int(pending_cols.shape[1])
                if cols.shape[1] >= need:
                    yield emit(
                        np.concatenate((pending_cols, cols[:, :need]), axis=1),
                        np.concatenate((pending_labels, labels[:need])),
                    )
                    cols = cols[:, need:]
                    labels = labels[need:]
                    pending_cols = None
                    pending_labels = None
                else:
                    pending_cols = np.concatenate((pending_cols, cols), axis=1)
                    pending_labels = np.concatenate((pending_labels, labels))
                    continue
            start = 0
            n_cols = int(cols.shape[1])
            while start + resolved_batch <= n_cols:
                stop = start + resolved_batch
                yield emit(cols[:, start:stop], labels[start:stop])
                start = stop
            if start < n_cols:
                pending_cols = cols[:, start:]
                pending_labels = labels[start:]
        if pending_cols is not None:
            assert pending_labels is not None
            yield emit(pending_cols, pending_labels)

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
        if not renormalize_subset:
            return super()._write_normalized_payload(
                cell_idx,
                feat_idx,
                location,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
                mirror=mirror,
            )

        from ..storage.materialize import write_renorm_subset_to_zarr

        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")
        if location not in self.z:
            self.z.create_group(location)
        if location + "/data" in self.z:
            return ChunkedArray(
                as_zarr_array(self.z[location + "/data"], name=location + "/data"),
                nthreads=self.nthreads,
                resources=self.resources,
            )

        write_renorm_subset_to_zarr(
            self,
            cell_idx,
            feat_idx,
            self.z,
            location + "/data",
            self.nthreads,
            log_transform=log_transform,
            mirror=mirror,
            stats_group=as_zarr_group(self.z[location], name=location),
        )
        return ChunkedArray(
            as_zarr_array(self.z[location + "/data"], name=location + "/data"),
            nthreads=self.nthreads,
            resources=self.resources,
        )

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        renormalize_subset: bool = False,
        log_transform: bool = False,
        **kwargs: Any,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data. Unlike the `normed` method in the generic Assay
        class this method is optimized for scRNA-Seq data and takes additional
        parameters that will be used by `norm_lib_size` (default normalization
        method for this class).

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix.
                      Defaults to the complete physical feature axis.
            renormalize_subset: If true, normalize using only ``feat_idx`` rather
                                than total expression across all features in a cell.
                                (Default value: False)
            log_transform: If True, then the normalized data is log-transformed (Default value: False).
            **kwargs: kwargs have no effect here.

        Returns:
            A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = np.arange(self.feats.N, dtype=np.int64)
        counts = self.rawData[:, feat_idx][cell_idx, :]
        norm_method_cache = self.normMethod
        scalar_cache = self.scalar
        try:
            if log_transform:
                self.normMethod = norm_lib_size_log
            if renormalize_subset:
                scalar = compute_with_progress(
                    counts.sum(axis=1),
                    "Normalizing with feature subset",
                    self.nthreads,
                )
                scalar[scalar == 0] = 1
                self.scalar = scalar
            else:
                self.scalar = self.cells.fetch_all(self.name + "_nCounts")[cell_idx]
            return self.normMethod(self, counts)
        finally:
            self.normMethod = norm_method_cache
            self.scalar = scalar_cache

    def _raw_feature_stream_source(self) -> tuple[zarr.Array, int, int]:
        """Return the preferred raw array and its feature and cell axes."""
        if self.rawDataT is not None:
            return self.rawDataT, 0, 1
        return cast(zarr.Array, self.rawData._backing), 1, 0

    def iter_raw_column_blocks(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int,
        msg: str | None = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, float, str], None, None]:
        """Read raw count column batches with shallow read-ahead."""
        yield from self._iter_raw_column_blocks(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            msg=msg,
        )

    def _iter_raw_column_blocks(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int | None,
        msg: str | None = None,
        *,
        plan: FeatureStreamPlan | None = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, float, str], None, None]:
        """Read raw count column batches with shallow read-ahead.

        Yields ``(block_idx, raw, feat_cols, read_sec, source)`` where ``raw`` has
        shape ``(len(cell_idx), len(feat_cols))``.
        """
        from ..utils.prefetch import iter_column_blocks

        cell_idx = np.asarray(cell_idx)
        feat_idx = np.asarray(feat_idx)
        zarr_arr, feature_axis, cell_axis = self._raw_feature_stream_source()
        if plan is None:
            raw_itemsize = max(1, int(np.dtype(zarr_arr.dtype).itemsize))
            plan = plan_feature_stream(
                zarr_arr,
                featureAxis=feature_axis,
                cellAxis=cell_axis,
                featureIndices=feat_idx,
                cellIndices=cell_idx,
                resources=self.resources,
                blockBytes=lambda width: max(
                    1,
                    len(cell_idx) * width * raw_itemsize,
                ),
                requestedBatchSize=batch_size,
            )
        batches = [block.indices for block in plan.blocks]
        n_blocks = len(plan.blocks)
        if msg:
            logger.debug(
                f"({self.name}) {msg}: {len(feat_idx)} features in "
                f"{n_blocks} geometry-planned blocks "
                f"(repeated chunk decodes={plan.repeatedDecodeCount})"
            )

        if feature_axis == 0:

            def read_block(block_idx: int) -> np.ndarray:
                return _read_facade_block(zarr_arr, batches[block_idx], cell_idx).T

        else:

            def read_block(block_idx: int) -> np.ndarray:
                return _read_facade_block(zarr_arr, cell_idx, batches[block_idx])

        for block_idx, raw, read_sec, source in iter_column_blocks(
            n_blocks,
            read_block,
            workers=plan.readWorkers,
            io_concurrency=plan.ioConcurrency,
            msg=msg,
        ):
            yield block_idx, raw, batches[block_idx], read_sec, source

    def iter_raw_feature_major_blocks(
        self,
        cell_idx: np.ndarray,
        plan: FeatureStreamPlan,
        msg: str | None = None,
    ) -> Generator[
        tuple[IndexBlock, np.ndarray, float, str],
        None,
        None,
    ]:
        """Yield C-contiguous ``(features, cells)`` raw count blocks."""
        from ..utils.prefetch import iter_column_blocks

        cell_idx = np.asarray(cell_idx)
        zarr_arr, feature_axis, _ = self._raw_feature_stream_source()
        if feature_axis != plan.featureAxis:
            raise ValueError("Feature stream plan does not match the raw source")
        blocks = plan.blocks

        if feature_axis == 0:

            def read_block(block_idx: int) -> np.ndarray:
                block = blocks[block_idx]
                return np.ascontiguousarray(
                    _read_facade_block(
                        zarr_arr,
                        block.indices,
                        cell_idx,
                    )
                )

        else:

            def read_block(block_idx: int) -> np.ndarray:
                block = blocks[block_idx]
                raw = _read_facade_block(
                    zarr_arr,
                    cell_idx,
                    block.indices,
                )
                return np.ascontiguousarray(raw.T)

        for block_idx, raw, read_sec, source in iter_column_blocks(
            len(blocks),
            read_block,
            workers=plan.readWorkers,
            io_concurrency=plan.ioConcurrency,
            msg=msg,
        ):
            yield blocks[block_idx], raw, read_sec, source
            del raw

    def iter_raw_feature_columns(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int,
        scalar: np.ndarray,
        sf: float,
        log_transform: bool = False,
        msg: str | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate library-size normalized feature columns."""
        yield from self._iter_raw_feature_columns(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            scalar=scalar,
            sf=sf,
            log_transform=log_transform,
            msg=msg,
        )

    def _iter_raw_feature_columns(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int | None,
        scalar: np.ndarray,
        sf: float,
        log_transform: bool = False,
        msg: str | None = None,
        *,
        plan: FeatureStreamPlan | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate library-size normalized feature columns without streaming
        the full normalized matrix.

        Raw count columns are read directly from the backing Zarr array in
        chunk-aligned batches and normalized in memory using a precomputed
        per-cell scalar (library size). Reads are prefetched in parallel.

        Args:
            cell_idx: Integer indices of cells to include (in output order).
            feat_idx: Integer indices of features to iterate over.
            batch_size: Number of feature columns per batch.
            scalar: Per-cell normalization factor aligned to ``cell_idx``.
            sf: Size factor multiplier applied before dividing by ``scalar``.
            log_transform: If True, apply ``log1p`` after normalization.
            msg: Progress bar description.

        Yields:
            Tuples of ``(normed_batch, feat_index_batch)`` where ``normed_batch``
            has shape ``(len(cell_idx), batch_columns)``.
        """
        import time

        from ..utils.process import process_rss_mb

        cell_idx = np.asarray(cell_idx)
        scalar_col = np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
        scalar_col[scalar_col == 0] = 1
        feat_idx = np.asarray(feat_idx)
        if plan is None:
            zarr_arr, feature_axis, cell_axis = self._raw_feature_stream_source()
            raw_itemsize = max(1, int(np.dtype(zarr_arr.dtype).itemsize))
            plan = plan_feature_stream(
                zarr_arr,
                featureAxis=feature_axis,
                cellAxis=cell_axis,
                featureIndices=feat_idx,
                cellIndices=cell_idx,
                resources=self.resources,
                blockBytes=lambda width: max(
                    1,
                    len(cell_idx)
                    * width
                    * (
                        raw_itemsize
                        + np.dtype(np.float32).itemsize
                        + np.dtype(np.float64).itemsize
                    ),
                ),
                requestedBatchSize=batch_size,
            )
        n_batches = len(plan.blocks)

        for block_idx, raw, cols, read_sec, source in self._iter_raw_column_blocks(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            msg=msg,
            plan=plan,
        ):
            t0 = time.perf_counter()
            normed = (sf * raw.astype(np.float32)) / scalar_col
            if log_transform:
                normed = np.log1p(normed)
            if msg:
                logger.debug(
                    f"({self.name}) {msg} batch {block_idx + 1}/{n_batches}: "
                    f"cols={len(cols)} read {read_sec:.1f}s ({source}) "
                    f"norm {time.perf_counter() - t0:.1f}s "
                    f"rss {process_rss_mb():.0f} MiB"
                )
            yield normed, cols

    def _mean_normed_feature_groups(
        self,
        cell_idx: np.ndarray,
        feature_groups: dict[str, np.ndarray],
        block_rows: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-cell mean of library-size normalized counts for each feature group.

        Reads the union of all requested feature columns once and streams over
        row blocks aligned to the array's on-disk row chunk. This avoids the
        full ChunkedArray normalization path (and
        its repeated wide-chunk reads) used by ``normed`` when scoring small,
        scattered gene sets such as cell cycle markers. Values are computed in
        float64 to match ``norm_lib_size``. Row blocks are read ahead in
        parallel and accumulated as they arrive (each writes a disjoint row
        slice, so order does not matter).
        """
        from ..storage.parallel import stream_shards

        zarr_arr = cast(zarr.Array, self.rawData._backing)
        cell_idx = np.asarray(cell_idx)
        if self.normMethod is norm_lib_size and self.sf is None:
            raise ValueError(
                "RNA library-size normalization requires a size factor (sf), got None"
            )
        sf = float(self.sf) if self.sf is not None else 1.0
        scalar = np.asarray(
            self.cells.fetch_all(self.name + "_nCounts")[cell_idx], dtype=np.float64
        )
        scalar[scalar == 0] = 1

        union = np.unique(
            np.concatenate([np.asarray(v, dtype=int) for v in feature_groups.values()])
        )
        local_pos = {
            key: np.searchsorted(union, np.asarray(idx, dtype=int))
            for key, idx in feature_groups.items()
        }

        n_cells = len(cell_idx)
        out = {key: np.empty(n_cells, dtype=np.float64) for key in feature_groups}
        if n_cells == 0:
            return out

        geometry = array_geometry(zarr_arr)
        if block_rows is None:
            block_rows = row_band(geometry, unit="chunk", fallback=n_cells)
        block_rows = max(1, int(block_rows))

        starts = range(0, n_cells, block_rows)

        def read(start: int) -> tuple[int, np.ndarray]:
            rows = cell_idx[start : start + block_rows]
            return start, _read_facade_block(zarr_arr, rows, union)

        block_bytes = (
            block_rows
            * max(1, len(union))
            * (np.dtype(zarr_arr.dtype).itemsize + np.dtype(np.float64).itemsize)
        )
        resident_bytes = (
            scalar.nbytes
            + union.nbytes
            + sum(value.nbytes for value in out.values())
            + sum(value.nbytes for value in local_pos.values())
        )
        admission = admit_stream(
            self.resources,
            nBlocks=self.resources.workers,
            blockBytes=block_bytes,
            decodeBytes=0 if geometry is None else geometry.nominalChunkBytes(),
            residentBytes=resident_bytes,
            requested=self.resources.workers,
        )
        for start, raw in stream_shards(
            starts,
            read,
            workers=admission.outerWorkers,
            io_concurrency=admission.ioConcurrency,
        ):
            end = start + raw.shape[0]
            normed = (sf * raw.astype(np.float64)) / scalar[start:end, None]
            for key, pos in local_pos.items():
                out[key][start:end] = normed[:, pos].mean(axis=1)
        return out

    def _streaming_feature_stats(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-feature library-size normalized stats via cell-band countsT.

        Reads each feature group by physical cell band, accumulates counts,
        sums, and squared sums in deterministic band order, and returns
        ``normed_tot``, ``normed_n``, and ``sigmas`` matching ``norm_lib_size``.
        """
        import time

        from ..storage.feature_stream import map_feature_cell_bands
        from ..utils.process import process_rss_mb

        cell_idx = np.asarray(cell_idx)
        feat_idx = np.asarray(feat_idx)
        if self.normMethod is norm_lib_size and self.sf is None:
            raise ValueError(
                "RNA library-size normalization requires a size factor (sf), got None"
            )
        sf = float(self.sf) if self.sf is not None else 1.0
        scalar = np.asarray(
            self.cells.fetch_all(self.name + "_nCounts")[cell_idx], dtype=np.float64
        )
        scalar[scalar == 0] = 1
        inv_scalar = 1.0 / scalar

        n_features = len(feat_idx)
        n_cells = len(cell_idx)
        nz = np.zeros(n_features, dtype=np.float64)
        s1 = np.zeros(n_features, dtype=np.float64)
        s2 = np.zeros(n_features, dtype=np.float64)
        if n_cells == 0 or n_features == 0:
            return {"normed_tot": s1, "normed_n": nz, "sigmas": s2}

        counts_t = self.rawDataT
        if counts_t is None:
            raise ValueError(
                f"RNA assay {self.name!r} requires sharded countsT "
                "for feature statistics"
            )
        n_feats = int(counts_t.shape[0])
        dest_of = np.full(n_feats, -1, dtype=np.int64)
        dest_of[feat_idx] = np.arange(n_features, dtype=np.int64)
        logger.info(
            f"({self.name}) feature stats consume "
            f"workers={self.resources.workers} "
            f"memoryBytes={self.resources.memoryBytes}"
        )

        from collections import defaultdict

        from ..utils.compute import add_stat_arrays, pairwise_merge_tree

        partials: dict[
            tuple[int, int],
            list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        ] = defaultdict(list)

        def process_band(
            band: Any,
        ) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray] | None:
            destinations = dest_of[band.featStart : band.featEnd]
            if not np.any(destinations >= 0):
                return None
            n_local = int(band.featEnd - band.featStart)
            local_nz = np.zeros(n_local, dtype=np.float64)
            local_s1 = np.zeros(n_local, dtype=np.float64)
            local_s2 = np.zeros(n_local, dtype=np.float64)
            local_dest = np.where(
                destinations >= 0,
                np.arange(n_local, dtype=np.int64),
                np.int64(-1),
            )
            t_compute = time.perf_counter()
            _hvg_stats_gene_major(
                band.values,
                inv_scalar[band.selectedDestinations],
                float(sf),
                local_dest,
                local_nz,
                local_s1,
                local_s2,
                selected=band.selectedLocal,
            )
            compute_sec = time.perf_counter() - t_compute
            logger.debug(
                f"({self.name}) feature stats band "
                f"{band.featStart}:{band.featEnd} cells "
                f"{band.cellStart}:{band.cellEnd}: "
                f"read {band.readSec:.1f}s compute {compute_sec:.1f}s "
                f"rss {process_rss_mb():.0f} MiB"
            )
            return (
                int(band.unitIndex),
                int(band.featStart),
                int(band.featEnd),
                local_nz,
                local_s1,
                local_s2,
            )

        consume_metrics: dict[str, object] = {}
        try:
            for item in map_feature_cell_bands(
                counts_t,
                process_band,
                cell_idx=cell_idx,
                feat_idx=feat_idx,
                resources=self.resources,
                progress="Calculating feature statistics",
                io=getattr(self, "storageIo", None),
                metrics=consume_metrics,
                orderedCompute=False,
            ):
                if item is None:
                    continue
                unit_index, feat_start, feat_end, local_nz, local_s1, local_s2 = item
                partials[(feat_start, feat_end)].append(
                    (unit_index, local_nz, local_s1, local_s2)
                )
            for (feat_start, feat_end), items in partials.items():
                items.sort(key=lambda row: row[0])
                merged = pairwise_merge_tree(
                    [(row[1], row[2], row[3]) for row in items],
                    add_stat_arrays,
                )
                destinations = dest_of[feat_start:feat_end]
                keep = destinations >= 0
                nz[destinations[keep]] = merged[0][keep]
                s1[destinations[keep]] = merged[1][keep]
                s2[destinations[keep]] = merged[2][keep]
        finally:
            if consume_metrics:
                logger.info(
                    f"({self.name}) feature stats execution "
                    f"read={consume_metrics.get('actualReadWorkers')} "
                    f"compute={consume_metrics.get('actualComputeWorkers')} "
                    f"fetch={consume_metrics.get('fetchSeconds')}s "
                    f"computeSec={consume_metrics.get('computeSeconds')}s"
                )

        mean = s1 / n_cells
        sigmas = s2 / n_cells - np.square(mean)
        return {"normed_tot": s1, "normed_n": nz, "sigmas": sigmas}

    def _compute_feature_summary(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute sufficient feature statistics without persisting metadata."""
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if len(cell_idx) == 0 or len(feat_idx) == 0:
            zeros = np.zeros(len(feat_idx), dtype=np.float64)
            return {
                "normed_tot": zeros.copy(),
                "normed_n": zeros.copy(),
                "sigmas": zeros.copy(),
            }
        if self.normMethod is norm_lib_size:
            return self._streaming_feature_stats(cell_idx, feat_idx)
        normed = self.normed(cell_idx, feat_idx)
        return {
            "normed_tot": np.asarray(
                compute_with_progress(
                    normed.sum(axis=0),
                    f"({self.name}) Computing normed_tot",
                    self.nthreads,
                ),
                dtype=np.float64,
            ),
            "normed_n": np.asarray(
                compute_with_progress(
                    (normed > 0).sum(axis=0),
                    f"({self.name}) Computing nCells",
                    self.nthreads,
                ),
                dtype=np.float64,
            ),
            "sigmas": np.asarray(
                compute_with_progress(
                    normed.var(axis=0),
                    f"({self.name}) Computing sigmas",
                    self.nthreads,
                ),
                dtype=np.float64,
            ),
        }

    def _select_hvgs(
        self,
        summary: Mapping[str, np.ndarray],
        *,
        n_selected: int,
        min_cells: int,
        max_cells: int | float,
        top_n: int,
        min_var: float,
        max_var: float,
        min_mean: float,
        max_mean: float,
        n_bins: int,
        lowess_frac: float,
        blacklist: str,
        keep_bounds: bool,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        feature_names: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an HVG mask and corrected variance from sufficient stats."""
        from ..features.variability import (
            fit_lowess,
            select_highly_variable_features,
        )

        normed_tot = np.asarray(summary["normed_tot"], dtype=np.float64)
        normed_n = np.asarray(summary["normed_n"], dtype=np.float64)
        sigmas = np.asarray(summary["sigmas"], dtype=np.float64)
        expected = (self.feats.N,)
        if any(values.shape != expected for values in (normed_tot, normed_n, sigmas)):
            raise ValueError(
                f"RNA feature-summary arrays must all have shape {expected}"
            )
        avg = (
            normed_tot / n_selected
            if n_selected > 0
            else np.zeros_like(normed_tot, dtype=np.float64)
        )
        nz_mean = np.divide(
            normed_tot,
            normed_n,
            out=np.zeros_like(normed_tot, dtype=np.float64),
            where=normed_n != 0,
        )
        positive = avg > 0
        corrected_variance = np.zeros(avg.shape, dtype=np.float64)
        if positive.any():
            corrected_variance[positive] = fit_lowess(
                avg[positive],
                sigmas[positive],
                n_bins,
                lowess_frac,
                bin_strategy=bin_strategy,
            )
        resolved_feature_names = (
            np.asarray(self.feats.fetch_all("names"))
            if feature_names is None
            else np.asarray(feature_names)
        )
        if resolved_feature_names.shape != expected:
            raise ValueError(
                f"RNA feature names must have shape {expected}, got "
                f"{resolved_feature_names.shape}"
            )
        values = select_highly_variable_features(
            corrected_variance=corrected_variance,
            normalized_cell_counts=normed_n,
            mean_nonzero=nz_mean,
            active_features=np.ones(self.feats.N, dtype=bool),
            feature_names=resolved_feature_names,
            min_cells=min_cells,
            max_cells=max_cells,
            top_n=top_n,
            min_var=min_var,
            max_var=max_var,
            min_mean=min_mean,
            max_mean=max_mean,
            blacklist=blacklist,
            keep_bounds=keep_bounds,
        )
        logger.info(f"{int(values.sum())} genes marked as HVGs")
        return np.asarray(values, dtype=bool), corrected_variance

    @staticmethod
    def _plot_hvgs(
        summary: Mapping[str, np.ndarray],
        values: np.ndarray,
        corrected_variance: np.ndarray,
        **plot_kwargs: Any,
    ) -> None:
        """Plot an artifact-backed HVG result without mounted feature stats."""
        from ..plotting import highly_variable_features

        normed_tot = np.asarray(summary["normed_tot"], dtype=np.float64)
        normed_n = np.asarray(summary["normed_n"], dtype=np.float64)
        nz_mean = np.divide(
            normed_tot,
            normed_n,
            out=np.zeros_like(normed_tot, dtype=np.float64),
            where=normed_n != 0,
        )
        highly_variable_features(
            mean_nonzero=nz_mean,
            corrected_variance=np.asarray(corrected_variance, dtype=np.float64),
            n_cells=normed_n,
            selected=np.asarray(values, dtype=bool),
            show=True,
            **plot_kwargs,
        )
