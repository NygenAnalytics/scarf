from collections.abc import Generator, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from numba import njit

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.execution import admit_stream
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.types import as_zarr_group
from ..utils.arguments import integer_argument
from ..utils.compute import compute_with_progress
from ..utils.logging import logger
from ..utils.moments import ColumnMoments, add_implicit_zeros, welford_add
from .base import Assay
from .normalization import (
    NORMALIZATION_PARAM_NAMES,
    _feature_group_positions,
    _feature_subset_source,
    check_normalization_flags,
    lib_size_feature_stream_eligible,
    library_size_divisors,
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
    norm_tf_idf,
    reject_unknown_normalization_params,
    uses_library_size_normalization,
)

# The Python objects that keep one band's partial feature statistics: three
# array headers, the ColumnMoments that holds two of them, the tuple that
# holds the moments, and its list slot (about 530 bytes).
_BAND_PARTIAL_BYTES = 640
# Four int64 index arrays and one mask over the feature rows of a band.
_BAND_ROW_INDEX_BYTES = 4 * 8 + 1


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
    out_m2: np.ndarray,
    log_transform: bool = False,
) -> None:
    """Write lib-size HVG statistics of the selected cells of a raw block.

    For each gene with a destination, writes the number of positive values,
    their sum, and their ``m2``, the sum of squared deviations from the mean
    over every selected cell. Stored counts update a running mean and ``m2``,
    and the cells without a count then join as zeros, so a gene whose values
    are all equal gets an ``m2`` of exactly zero.
    """
    n_genes = values.shape[0]
    n_selected = selected.shape[0]
    for g in range(n_genes):
        target = dest[g]
        if target < 0:
            continue
        c_nz = 0.0
        c_s1 = 0.0
        stored = 0
        mean = 0.0
        m2 = 0.0
        for i in range(n_selected):
            count = values[g, selected[i]]
            if count == 0:
                continue
            value = sf * np.float64(count) * inv[i]
            if log_transform:
                value = np.log1p(value)
            if value > 0.0:
                c_nz += 1.0
            c_s1 += value
            stored += 1
            mean, m2 = welford_add(stored, mean, m2, value)
        out_nz[target] = c_nz
        out_s1[target] = c_s1
        out_m2[target] = add_implicit_zeros(m2, mean, stored, n_selected)


def _hvg_stats_gene_major(
    values: np.ndarray,
    inv: np.ndarray,
    sf: float,
    dest: np.ndarray,
    out_nz: np.ndarray,
    out_s1: np.ndarray,
    out_m2: np.ndarray,
    selected: np.ndarray | None = None,
    log_transform: bool = False,
) -> None:
    """Write lib-size HVG statistics of a gene-major count block.

    ``selected`` holds the block columns of the cells to summarize, every
    column by default, and ``inv`` their inverse library totals. Each gene
    with a destination gets the number of positive values, their sum, and
    their ``m2`` over the selected cells; ``ColumnMoments.merge`` combines
    the sums and ``m2`` of different blocks.
    """
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
        out_m2,
        log_transform,
    )


def _merge_band_statistics(
    first: tuple[np.ndarray, ColumnMoments],
    second: tuple[np.ndarray, ColumnMoments],
) -> tuple[np.ndarray, ColumnMoments]:
    """Merge the detections and moments of two cell bands of a feature group."""
    return first[0] + second[0], first[1].merge(second[1])


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
            if not self.z.read_only:
                self.attrs["size_factor"] = self.sf
        self.scalar: np.ndarray | None = None

    requiresCountsT = True

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
        reject_unknown_normalization_params(
            norm_params, caller="iter_normed_feature_wise"
        )
        log_transform, renormalize_subset = check_normalization_flags(
            self,
            log_transform=norm_params.get("log_transform", False),
            renormalize_subset=norm_params.get("renormalize_subset", False),
        )
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
                scratch_itemsize=scratch_itemsize,
                resident_bytes=resident_bytes,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
            )
            return

        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if cell_idx.ndim != 1 or feat_idx.ndim != 1:
            raise ValueError("cell_idx and feat_idx must be one-dimensional")
        scratch_itemsize = integer_argument(
            scratch_itemsize, "scratch_itemsize", minimum=0
        )
        resident_bytes = integer_argument(resident_bytes, "resident_bytes", minimum=0)

        if msg is None:
            msg = ""

        # Eligibility requires a size factor.
        sf = self.sf
        assert sf is not None
        if feat_idx.size == 0:
            return
        scalar_values = library_size_divisors(
            self._cell_count_totals(cell_idx), source=self._totals_name, copy=False
        )
        counts_t = self.rawDataT
        # An RNAassay cannot be constructed without a complete countsT.
        assert counts_t is not None
        n_feats = int(counts_t.shape[0])
        dest_of = np.full(n_feats, -1, dtype=np.int64)
        dest_of[feat_idx] = np.arange(len(feat_idx), dtype=np.int64)
        feat_labels = np.asarray(feat_idx)
        from ..storage.feature_stream import (
            map_feature_read_groups,
            persisted_read_group,
            read_group_stream_floor,
        )

        n_cells = int(cell_idx.shape[0])
        float64_size = int(np.dtype(np.float64).itemsize)
        raw_size = int(np.dtype(counts_t.dtype).itemsize)
        # Per emitted value: the float64 batch being filled plus, while it is
        # filled, the previous batch still held by the caller and one raw
        # copy. While the caller processes a batch it needs the batch and its
        # declared scratch instead.
        item_bytes = float64_size + max(float64_size + raw_size, scratch_itemsize)
        feature_width, _ = persisted_read_group(counts_t)
        resident = resident_bytes + scalar_values.nbytes + dest_of.nbytes
        # The stream keeps at least one read group and one band read.
        available = (
            self.resources.memoryBytes
            - resident
            - read_group_stream_floor(counts_t, cell_idx=cell_idx, feat_idx=feat_idx)
        )
        affordable = max(0, available) // max(1, n_cells * item_bytes)
        if batch_size is None:
            width = min(affordable, feature_width, max(1, len(feat_idx)))
        else:
            width = max(1, int(batch_size))
        if width < 1 or width > affordable:
            raise MemoryError(
                f"Normalized feature batches of {max(1, width)} features over "
                f"{n_cells} cells do not fit the memory budget of "
                f"{self.resources.memoryBytes} bytes; the affordable width is "
                f"{affordable}. Increase the memory budget"
                + ("." if batch_size is None else " or reduce the batch size.")
            )
        loaded_groups = map_feature_read_groups(
            counts_t,
            lambda loaded: loaded,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=self.resources,
            progress=msg or None,
            io=getattr(self, "storageIo", None),
            scratchBytes=resident + width * n_cells * item_bytes,
        )

        def emit(
            values: np.ndarray, labels: np.ndarray
        ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
            # ``values`` is a C-order features-by-cells float64 block.
            if as_dataframe:
                return pd.DataFrame(values.T, columns=labels, copy=False)
            return values, labels

        block: np.ndarray | None = None
        block_labels = np.empty(0, dtype=feat_labels.dtype)
        filled = 0
        for group in loaded_groups:
            local_dest = dest_of[group.featStart : group.featEnd]
            rows = np.flatnonzero(local_dest >= 0)
            start = 0
            while start < rows.size:
                if block is None:
                    block = np.empty((width, n_cells), dtype=np.float64)
                    block_labels = np.empty(width, dtype=feat_labels.dtype)
                    filled = 0
                take = min(width - filled, int(rows.size) - start)
                piece = rows[start : start + take]
                # The float64 arithmetic of ``normed``, written into the batch.
                target = block[filled : filled + take]
                np.multiply(
                    group.values[piece], float(sf), out=target, dtype=np.float64
                )
                target /= scalar_values
                if log_transform:
                    np.log1p(target, out=target)
                block_labels[filled : filled + take] = feat_labels[local_dest[piece]]
                filled += take
                start += take
                if filled == width:
                    yield emit(block, block_labels)
                    block = None
            if batch_size is None and block is not None:
                yield emit(block[:filled], block_labels[:filled])
                block = None
            # The stream frees a read group once the next one is requested.
            del group
        if block is not None:
            yield emit(block[:filled], block_labels[:filled])

    def _write_normalized_payload(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        location: str,
        *,
        log_transform: bool,
        renormalize_subset: bool,
        mirror: zarr.Array | None = None,
    ) -> None:
        # The subset writer computes library-size values from the counts
        # itself, so every other normalizer is written from ``normed``.
        if not (renormalize_subset and uses_library_size_normalization(self)):
            super()._write_normalized_payload(
                cell_idx,
                feat_idx,
                location,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
                mirror=mirror,
            )
            return

        from .normalization import write_renorm_subset_to_zarr

        cell_idx, feat_idx = self._payload_indices(cell_idx, feat_idx)
        # Normalized values can be graph coordinates, so the payload is
        # written with a checked writer, as the generic payload is.
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
            requireFinite=True,
            operation="run_normalization",
        )

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        renormalize_subset: bool = False,
        log_transform: bool = False,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data. Unlike the `normed` method in the generic Assay
        class this method is optimized for scRNA-Seq data and supports
        ``renormalize_subset`` with `norm_lib_size` (default normalization
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
            log_transform: If True, then the normalized data is log-transformed
                           (Default value: False).

        Returns:
            A chunked array (delayed matrix) containing normalized data.
        """
        from ..storage.identity import read_dataset_fingerprint

        log_transform, renormalize_subset = check_normalization_flags(
            self,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
        )
        method = self.normMethod
        library_size = method is norm_lib_size or method is norm_lib_size_log
        if library_size and self.sf is None:
            raise ValueError(
                "RNA library-size normalization requires a size factor (sf), got None"
            )
        read_dataset_fingerprint(self.z)
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = np.arange(self.feats.N, dtype=np.int64)
        counts = self.rawData[:, feat_idx][cell_idx, :]
        if renormalize_subset:
            scalar = np.asarray(
                compute_with_progress(
                    counts.sum(axis=1, dtype=np.float64),
                    "Normalizing with feature subset",
                    self.nthreads,
                ),
                dtype=np.float64,
            )
            source = _feature_subset_source(self)
        else:
            scalar = self._cell_count_totals(cell_idx)
            source = self._totals_name
        if library_size:
            # Zero-total cells normalize to zero, as on every library-size
            # path, and invalid totals raise.
            scalar = library_size_divisors(scalar, source=source, copy=False)
        else:
            # Another normalization need not read the totals, so they are not
            # checked; one that reads them finds a zero total replaced by 1.
            scalar[scalar == 0] = 1
        # The method reads the totals from ``self.scalar`` while it builds the
        # lazy result, so concurrent calls must not interleave here.
        with self._normalization_lock:
            scalar_cache = self.scalar
            self.scalar = scalar
            try:
                values = method(self, counts)
            finally:
                self.scalar = scalar_cache
        if log_transform:
            # NumPy logs uint8 in float16 and uint16 in float32, so the
            # logarithms are taken in float64 for every value dtype.
            values = cast(ChunkedArray, np.log1p(values, dtype=np.float64))
        return values

    def _normalization_flags(self) -> frozenset[str]:
        """Return the normalization flags that ``normed`` applies.

        ``normed`` hands every normalizer each cell's total in ``scalar`` and
        can take ``log1p`` of its output. Library-size and custom normalizers
        take both flags. ``norm_lib_size_log`` values are already logarithms,
        so it takes only ``renormalize_subset``. ``norm_dummy`` reads no
        totals, so it takes only ``log_transform``. CLR values are log ratios
        and TF-IDF values are never logged, and neither reads the totals.
        """
        method = self.normMethod
        if method is norm_lib_size_log:
            return frozenset({"renormalize_subset"})
        if method is norm_dummy:
            return frozenset({"log_transform"})
        if method is norm_clr or method is norm_tf_idf:
            return frozenset()
        return NORMALIZATION_PARAM_NAMES

    def _mean_normed_feature_groups(
        self,
        cell_idx: np.ndarray,
        feature_groups: dict[str, np.ndarray],
        *,
        log_transform: bool = False,
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
        cell_idx = np.asarray(cell_idx)
        if (self.normMethod is norm_lib_size or log_transform) and self.sf is None:
            raise ValueError(
                "RNA library-size normalization requires a size factor (sf), got None"
            )
        sf = float(self.sf) if self.sf is not None else 1.0
        scalar = library_size_divisors(
            self._cell_count_totals(cell_idx), source=self._totals_name, copy=False
        )

        union = np.unique(
            np.concatenate([np.asarray(v, dtype=int) for v in feature_groups.values()])
        )
        local_pos = {
            key: np.searchsorted(union, np.asarray(idx, dtype=int))
            for key, idx in feature_groups.items()
        }
        return self._mean_normed_union(
            cell_idx,
            scalar,
            union,
            local_pos,
            sf=sf,
            log_transform=log_transform,
            resident_bytes=(
                scalar.nbytes
                + union.nbytes
                + sum(value.nbytes for value in local_pos.values())
            ),
        )

    def _mean_normed_union(
        self,
        cell_idx: np.ndarray,
        scalar: np.ndarray,
        union: np.ndarray,
        local_pos: Mapping[str, np.ndarray],
        *,
        sf: float,
        log_transform: bool,
        resident_bytes: int,
    ) -> dict[str, np.ndarray]:
        """Average library-size normalized ``union`` columns per group position.

        ``scalar`` holds the library-size divisor of each cell in ``cell_idx``.
        ``resident_bytes`` counts the arrays the caller holds for the call,
        including ``scalar``, ``union``, and ``local_pos``. Cells are read in
        blocks of one on-disk row chunk.
        """
        from ..storage.parallel import stream_shards

        zarr_arr = cast(zarr.Array, self.rawData._backing)
        n_cells = len(cell_idx)
        out = {key: np.empty(n_cells, dtype=np.float64) for key in local_pos}
        if n_cells == 0:
            return out

        geometry = array_geometry(zarr_arr)
        block_rows = row_band(geometry, unit="chunk", fallback=n_cells)

        starts = range(0, n_cells, block_rows)

        def read(start: int) -> tuple[int, np.ndarray]:
            rows = cell_idx[start : start + block_rows]
            return start, _read_facade_block(zarr_arr, rows, union)

        block_bytes = (
            block_rows
            * max(1, len(union))
            * (np.dtype(zarr_arr.dtype).itemsize + np.dtype(np.float64).itemsize)
        )
        admission = admit_stream(
            self.resources,
            nBlocks=self.resources.workers,
            blockBytes=block_bytes,
            decodeBytes=0 if geometry is None else geometry.nominalChunkBytes(),
            residentBytes=resident_bytes + sum(value.nbytes for value in out.values()),
            requested=self.resources.workers,
        )
        for start, raw in stream_shards(
            starts,
            read,
            workers=admission.readWorkers,
            io_concurrency=admission.ioConcurrency,
        ):
            end = start + raw.shape[0]
            normed = (sf * raw.astype(np.float64)) / scalar[start:end, None]
            if log_transform:
                np.log1p(normed, out=normed)
            for key, pos in local_pos.items():
                out[key][start:end] = normed[:, pos].mean(axis=1)
        return out

    def _iter_feature_group_means(
        self,
        cell_idx: np.ndarray,
        feature_groups: Sequence[np.ndarray],
        *,
        block_rows: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield per-cell group means, as ``iter_feature_group_means`` does.

        Library-size normalization depends only on each cell's total, so the
        totals are read once and each row band of ``block_rows`` cells reads
        the union of the group features once. Other normalizations use the
        fitted generic kernel.
        """
        if not lib_size_feature_stream_eligible(self):
            yield from super()._iter_feature_group_means(cell_idx, feature_groups)
            return
        assert self.sf is not None
        # Rejects empty or malformed groups exactly as the generic kernel does.
        union, positions = _feature_group_positions(feature_groups)
        keyed = {str(index): position for index, position in enumerate(positions)}
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        totals = library_size_divisors(
            self._cell_count_totals(cell_idx), source=self._totals_name, copy=False
        )
        resident_bytes = (
            totals.nbytes + union.nbytes + sum(value.nbytes for value in positions)
        )
        band = max(1, len(cell_idx) if block_rows is None else int(block_rows))
        for start in range(0, len(cell_idx), band):
            means = self._mean_normed_union(
                cell_idx[start : start + band],
                totals[start : start + band],
                union,
                keyed,
                sf=float(self.sf),
                log_transform=False,
                resident_bytes=resident_bytes,
            )
            yield np.column_stack([means[key] for key in keyed])

    def _streaming_feature_stats(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        *,
        log_transform: bool = False,
    ) -> dict[str, np.ndarray]:
        """Per-feature library-size normalized stats via cell-band countsT.

        Reads each feature group by physical cell band, computes each band's
        detections, sums, and sums of squared deviations from the band mean
        (``m2``), and merges the bands of a group in a fixed pairwise order
        with ``ColumnMoments.merge``. Returns ``normed_tot``, ``normed_n``,
        and ``sigmas``, the population variance ``m2 / n_cells``, matching
        ``norm_lib_size``. A feature whose normalized values are all equal
        has a variance of exactly zero when its band sums are exact.
        """
        import time

        from ..storage.feature_stream import (
            map_feature_cell_bands,
            persisted_read_group,
            selected_feature_chunk_starts,
        )
        from ..utils.process import rss_text

        cell_idx = np.asarray(cell_idx)
        feat_idx = np.asarray(feat_idx)
        if (self.normMethod is norm_lib_size or log_transform) and self.sf is None:
            raise ValueError(
                "RNA library-size normalization requires a size factor (sf), got None"
            )
        sf = float(self.sf) if self.sf is not None else 1.0
        inv_scalar = library_size_divisors(
            self._cell_count_totals(cell_idx), source=self._totals_name, copy=False
        )
        np.reciprocal(inv_scalar, out=inv_scalar)

        n_features = len(feat_idx)
        n_cells = len(cell_idx)
        nz = np.zeros(n_features, dtype=np.float64)
        s1 = np.zeros(n_features, dtype=np.float64)
        m2 = np.zeros(n_features, dtype=np.float64)
        if n_cells == 0 or n_features == 0:
            return {"normed_tot": s1, "normed_n": nz, "sigmas": m2}

        # An RNA assay opens only with a complete countsT.
        counts_t = self.rawDataT
        assert counts_t is not None
        n_feats = int(counts_t.shape[0])
        dest_of = np.full(n_feats, -1, dtype=np.int64)
        dest_of[feat_idx] = np.arange(n_features, dtype=np.int64)
        logger.info(
            f"({self.name}) feature stats consume "
            f"workers={self.resources.workers} "
            f"memoryBytes={self.resources.memoryBytes}"
        )
        feat_chunk, cell_chunk = (int(extent) for extent in counts_t.chunks)
        group_starts = selected_feature_chunk_starts(counts_t, feat_idx)
        group_rows = sum(min(feat_chunk, n_feats - start) for start in group_starts)
        n_bands = min(n_cells, -(-int(counts_t.shape[1]) // cell_chunk))
        widest = min(n_feats, max(feat_chunk, persisted_read_group(counts_t)[0]))
        # The inverse totals, the feature destinations, and the outputs stay for
        # the whole stream, and every band keeps three float64 partial
        # statistics per feature row of its group until the stream ends. Each
        # compute worker also gathers the inverse totals of a band's cells and
        # indexes its feature rows.
        scratch_bytes = (
            inv_scalar.nbytes
            + dest_of.nbytes
            + nz.nbytes
            + s1.nbytes
            + m2.nbytes
            + n_bands
            * (3 * nz.itemsize * group_rows + len(group_starts) * _BAND_PARTIAL_BYTES)
            + self.resources.workers
            * (
                min(cell_chunk, n_cells) * inv_scalar.itemsize
                + widest * _BAND_ROW_INDEX_BYTES
            )
        )

        from collections import defaultdict

        from ..utils.compute import pairwise_merge_tree

        partials: dict[
            tuple[int, int],
            list[tuple[int, np.ndarray, ColumnMoments]],
        ] = defaultdict(list)

        # Every feature group holds a selected feature, so every band does.
        def process_band(
            band: Any,
        ) -> tuple[int, int, int, np.ndarray, ColumnMoments]:
            rows = band.featureRows()
            destinations = dest_of[band.featStart + rows]
            n_local = int(band.featEnd - band.featStart)
            local_nz = np.zeros(n_local, dtype=np.float64)
            local_s1 = np.zeros(n_local, dtype=np.float64)
            local_m2 = np.zeros(n_local, dtype=np.float64)
            local_dest = np.where(destinations >= 0, rows, np.int64(-1))
            t_compute = time.perf_counter()
            _hvg_stats_gene_major(
                band.values,
                inv_scalar[band.selectedDestinations],
                float(sf),
                local_dest,
                local_nz,
                local_s1,
                local_m2,
                selected=band.selectedLocal,
                log_transform=log_transform,
            )
            compute_sec = time.perf_counter() - t_compute
            logger.opt(lazy=True).debug(
                f"({self.name}) feature stats band "
                f"{band.featStart}:{band.featEnd} cells "
                f"{band.cellStart}:{band.cellEnd}: "
                f"read {band.readSec:.1f}s compute {compute_sec:.1f}s "
                "rss {rss}",
                rss=rss_text,
            )
            return (
                int(band.unitIndex),
                int(band.featStart),
                int(band.featEnd),
                local_nz,
                ColumnMoments(len(band.selectedLocal), local_s1, local_m2),
            )

        consume_metrics: dict[str, object] = {}
        # The stream fills the metrics when it is created, so every stream
        # that starts is logged however it ends.
        bands = map_feature_cell_bands(
            counts_t,
            process_band,
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            resources=self.resources,
            progress="Calculating feature statistics",
            io=getattr(self, "storageIo", None),
            metrics=consume_metrics,
            scratchBytes=scratch_bytes,
            orderedCompute=False,
        )
        try:
            for item in bands:
                unit_index, feat_start, feat_end, local_nz, local_moments = item
                partials[(feat_start, feat_end)].append(
                    (unit_index, local_nz, local_moments)
                )
            while partials:
                (feat_start, feat_end), items = partials.popitem()
                items.sort(key=lambda row: row[0])
                detected, moments = pairwise_merge_tree(
                    [(row[1], row[2]) for row in items],
                    _merge_band_statistics,
                )
                if moments.count != n_cells:
                    raise RuntimeError(
                        f"Feature statistics of features {feat_start}:{feat_end} "
                        f"covered {moments.count} of {n_cells} selected cells"
                    )
                destinations = dest_of[feat_start:feat_end]
                keep = destinations >= 0
                nz[destinations[keep]] = detected[keep]
                s1[destinations[keep]] = moments.total[keep]
                m2[destinations[keep]] = moments.m2[keep]
        finally:
            logger.info(
                f"({self.name}) feature stats execution "
                f"read={consume_metrics.get('actualReadWorkers')} "
                f"compute={consume_metrics.get('actualComputeWorkers')} "
                f"fetch={consume_metrics.get('fetchSeconds')}s "
                f"computeSec={consume_metrics.get('computeSeconds')}s"
            )

        m2 /= n_cells
        return {"normed_tot": s1, "normed_n": nz, "sigmas": m2}

    def _compute_feature_summary(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        *,
        log_transform: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute sufficient feature statistics without persisting metadata.

        ``log_transform`` summarizes the values of
        ``normed(log_transform=True)``.
        """
        log_transform, _ = check_normalization_flags(
            self, log_transform=log_transform, renormalize_subset=False
        )
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if len(cell_idx) == 0 or len(feat_idx) == 0:
            zeros = np.zeros(len(feat_idx), dtype=np.float64)
            return {
                "normed_tot": zeros.copy(),
                "normed_n": zeros.copy(),
                "sigmas": zeros.copy(),
            }
        if uses_library_size_normalization(self):
            return self._streaming_feature_stats(
                cell_idx, feat_idx, log_transform=log_transform
            )
        normed = self.normed(cell_idx, feat_idx, log_transform=log_transform)
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
        feature_names: np.ndarray,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an HVG mask and corrected variance from sufficient stats.

        ``summary`` is a feature summary of this assay and ``feature_names``
        holds one name per feature; ``select_highly_variable_features``
        rejects inputs of any other length.
        """
        from ..features.variability import (
            fit_lowess,
            select_highly_variable_features,
        )

        normed_tot = np.asarray(summary["normed_tot"], dtype=np.float64)
        normed_n = np.asarray(summary["normed_n"], dtype=np.float64)
        sigmas = np.asarray(summary["sigmas"], dtype=np.float64)
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
        values = select_highly_variable_features(
            corrected_variance=corrected_variance,
            normalized_cell_counts=normed_n,
            mean_nonzero=nz_mean,
            active_features=np.ones(self.feats.N, dtype=bool),
            feature_names=np.asarray(feature_names),
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
