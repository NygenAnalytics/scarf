from collections.abc import Generator, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from ..storage.types import as_zarr_array, as_zarr_group
from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..utils.compute import show_dask_progress
from ..utils.logging import logger
from .base import Assay
from .normalization import (
    lib_size_feature_stream_eligible,
    norm_lib_size,
    norm_lib_size_log,
)
from .persistence import _feature_stats_tile_shape


def _read_facade_block(
    zarr_arr: zarr.Array,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    from . import _read_block

    return _read_block(zarr_arr, row_idx, col_idx)


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

    def __init__(
        self,
        z: zarr.Group,
        name: str,
        cell_data: MetaData,
        *,
        workspace: str | None = None,
        nthreads: int = 1,
        min_cells_per_feature: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            z=z,
            workspace=workspace,
            name=name,
            cell_data=cell_data,
            nthreads=nthreads,
            min_cells_per_feature=min_cells_per_feature,
            **kwargs,
        )
        self.normMethod = norm_lib_size
        if "size_factor" in self.attrs:
            self.sf = int(cast(int, self.attrs["size_factor"]))
        else:
            self.sf = 1000
            self.attrs["size_factor"] = self.sf
        self.scalar: np.ndarray | None = None

    def iter_normed_feature_wise(
        self,
        cell_key: str | None,
        feat_key: str | None,
        batch_size: int,
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
                cell_key,
                feat_key,
                batch_size,
                msg,
                as_dataframe=as_dataframe,
                **norm_params,
            )
            return

        if cell_key is None:
            cell_idx = np.array(list(range(self.cells.N)))
        else:
            cell_idx = self.cells.active_index(cell_key)

        if feat_key is None:
            feat_idx = np.array(list(range(self.feats.N)))
        else:
            feat_idx = self.feats.active_index(feat_key)

        if msg is None:
            msg = ""

        sf = self.sf
        if sf is None:
            raise ValueError("RNA library-size normalization requires a size factor")
        scalar = self.cells.fetch_all(self.name + "_nCounts")[cell_idx]
        log_transform = bool(norm_params.get("log_transform", False))
        for mat, cols in self.iter_raw_feature_columns(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            scalar=scalar,
            sf=float(sf),
            log_transform=log_transform,
            msg=msg,
        ):
            mat64 = np.asarray(mat, dtype=np.float64)
            if as_dataframe:
                yield pd.DataFrame(mat64, columns=cols)
            else:
                yield mat64.T, cols

    def save_normalized_data(
        self,
        cell_key: str,
        feat_key: str,
        batch_size: int,
        location: str,
        log_transform: bool,
        renormalize_subset: bool,
        update_keys: bool,
        mirror: zarr.Array | None = None,
    ) -> ChunkedArray:
        if not renormalize_subset:
            return super().save_normalized_data(
                cell_key,
                feat_key,
                batch_size,
                location,
                log_transform,
                renormalize_subset,
                update_keys,
                mirror=mirror,
            )

        from ..storage.materialize import write_renorm_subset_to_zarr

        if feat_key != "I":
            feat_key = cell_key + "__" + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        }
        if location in self.z:
            attrs = self.z[location].attrs
            if (
                attrs.get("subset_hash") == subset_hash
                and attrs.get("subset_params") == subset_params
            ):
                logger.info(
                    f"Using existing normalized data with cell key {cell_key} and feat key {feat_key}"
                )
                if update_keys:
                    self.attrs["latest_feat_key"] = (
                        feat_key.split("__", 1)[1] if feat_key != "I" else "I"
                    )
                    self.attrs["latest_cell_key"] = cell_key
                return ChunkedArray(
                    as_zarr_array(self.z[location + "/data"], name=location + "/data"),
                    nthreads=self.nthreads,
                )
            self.z.create_group(location, overwrite=True)

        write_renorm_subset_to_zarr(
            self,
            cell_idx,
            feat_idx,
            self.z,
            location + "/data",
            self.nthreads,
            log_transform=log_transform,
            mirror=mirror,
        )
        self.z[location].attrs["subset_hash"] = subset_hash
        self.z[location].attrs["subset_params"] = subset_params
        self._finalize_staged_mirror(mirror, subset_hash, subset_params)
        if update_keys:
            self.attrs["latest_feat_key"] = (
                feat_key.split("__", 1)[1] if feat_key != "I" else "I"
            )
            self.attrs["latest_cell_key"] = cell_key
        return ChunkedArray(
            as_zarr_array(self.z[location + "/data"], name=location + "/data"),
            nthreads=self.nthreads,
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
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell
                                (Default value: False)
            log_transform: If True, then the normalized data is log-transformed (Default value: False).
            **kwargs: kwargs have no effect here.

        Returns:
            A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        counts = self.rawData[:, feat_idx][cell_idx, :]
        norm_method_cache = self.normMethod
        if log_transform:
            self.normMethod = norm_lib_size_log
        if renormalize_subset:
            a = show_dask_progress(
                counts.sum(axis=1), "Normalizing with feature subset", self.nthreads
            )
            a[a == 0] = 1
            self.scalar = a
        else:
            self.scalar = self.cells.fetch_all(self.name + "_nCounts")[cell_idx]
        val = self.normMethod(self, counts)
        self.normMethod = norm_method_cache
        return val

    def iter_raw_column_blocks(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int,
        msg: str | None = None,
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, float, str], None, None]:
        """Read raw count column batches with remote-aware staging.

        Yields ``(block_idx, raw, feat_cols, read_sec, source)`` where ``raw`` has
        shape ``(len(cell_idx), len(feat_cols))``.
        """
        from ..storage.stores import is_remote_datastore
        from ..utils.prefetch import iter_column_blocks

        cell_idx = np.asarray(cell_idx)
        feat_idx = np.asarray(feat_idx)
        batch_size = max(1, batch_size)
        batches = [
            feat_idx[s : s + batch_size] for s in range(0, len(feat_idx), batch_size)
        ]
        n_blocks = len(batches)
        if msg:
            logger.debug(
                f"({self.name}) {msg}: {len(feat_idx)} features in "
                f"{n_blocks} batches (width {batch_size})"
            )

        use_counts_t = self.rawDataT is not None
        if use_counts_t:
            zarr_arr = cast(zarr.Array, self.rawDataT)

            def read_block(block_idx: int) -> np.ndarray:
                return _read_facade_block(zarr_arr, batches[block_idx], cell_idx).T

        else:
            zarr_arr = cast(zarr.Array, self.rawData._backing)

            def read_block(block_idx: int) -> np.ndarray:
                return _read_facade_block(zarr_arr, cell_idx, batches[block_idx])

        remote = is_remote_datastore(None, self.z)
        for block_idx, raw, read_sec, source in iter_column_blocks(
            n_blocks,
            read_block,
            remote=remote,
            msg=msg,
        ):
            yield block_idx, raw, batches[block_idx], read_sec, source

    def iter_raw_feature_columns(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        batch_size: int,
        scalar: np.ndarray,
        sf: float,
        log_transform: bool = False,
        prefetch_depth: int = 1,
        msg: str | None = None,
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
            prefetch_depth: Number of batches to read ahead in parallel.
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
        n_batches = max(
            1, (len(feat_idx) + max(1, batch_size) - 1) // max(1, batch_size)
        )

        for block_idx, raw, cols, read_sec, source in self.iter_raw_column_blocks(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            batch_size=batch_size,
            msg=msg,
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
        from ..storage.budget import worker_prefetch_depth
        from ..utils.prefetch import prefetch_blocks

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

        if block_rows is None:
            chunks = getattr(zarr_arr, "chunks", None)
            block_rows = int(chunks[0]) if chunks else n_cells
        block_rows = max(1, int(block_rows))

        starts = range(0, n_cells, block_rows)

        def read(start: int) -> tuple[int, np.ndarray]:
            rows = cell_idx[start : start + block_rows]
            return start, _read_facade_block(zarr_arr, rows, union)

        max_ahead = worker_prefetch_depth()
        for start, raw in prefetch_blocks(starts, read, max_ahead=max_ahead):
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
        """Per-feature library-size normalized stats in one streaming pass.

        Decodes each physical Zarr chunk at most once for the selected cells and
        features, normalizes dense sub-tiles in float64 in place, and accumulates
        per-feature nonzero count, sum, and sum of squares. Remote stores may
        prefetch the next physical chunk while compute continues. Values match
        ``norm_lib_size``. Returns ``normed_tot`` (sum), ``normed_n`` (nonzero
        count), and ``sigmas`` (population variance).
        """
        import time

        from ..storage.stores import is_remote_datastore
        from ..utils.prefetch import iter_column_blocks
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

        use_counts_t = self.rawDataT is not None
        if use_counts_t:
            zarr_arr = cast(zarr.Array, self.rawDataT)
            chunks = getattr(zarr_arr, "chunks", None)
            feat_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else n_features
            cell_chunk = int(chunks[1]) if chunks and len(chunks) > 1 else n_cells
        else:
            zarr_arr = cast(zarr.Array, self.rawData._backing)
            chunks = getattr(zarr_arr, "chunks", None)
            cell_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else n_cells
            feat_chunk = int(chunks[1]) if chunks and len(chunks) > 1 else n_features
        cell_chunk = max(1, cell_chunk)
        feat_chunk = max(1, feat_chunk)

        cell_pos = np.arange(n_cells, dtype=np.intp)
        feat_pos = np.arange(n_features, dtype=np.intp)
        cell_bins = np.asarray(cell_idx // cell_chunk, dtype=np.intp)
        feat_bins = np.asarray(feat_idx // feat_chunk, dtype=np.intp)

        tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if use_counts_t:
            for feat_bin in np.unique(feat_bins):
                feat_mask = feat_bins == feat_bin
                local_feats = feat_pos[feat_mask]
                cols = feat_idx[feat_mask]
                for cell_bin in np.unique(cell_bins):
                    cell_mask = cell_bins == cell_bin
                    local_cells = cell_pos[cell_mask]
                    rows = cell_idx[cell_mask]
                    tiles.append((local_cells, rows, local_feats, cols))
        else:
            for cell_bin in np.unique(cell_bins):
                cell_mask = cell_bins == cell_bin
                local_cells = cell_pos[cell_mask]
                rows = cell_idx[cell_mask]
                for feat_bin in np.unique(feat_bins):
                    feat_mask = feat_bins == feat_bin
                    local_feats = feat_pos[feat_mask]
                    cols = feat_idx[feat_mask]
                    tiles.append((local_cells, rows, local_feats, cols))

        n_blocks = len(tiles)
        remote = is_remote_datastore(None, self.z)
        sub_rows, sub_cols = _feature_stats_tile_shape(
            max((len(rows) for _, rows, _, _ in tiles), default=1),
            max((len(cols) for _, _, _, cols in tiles), default=1),
            row_chunk=cell_chunk,
            col_chunk=feat_chunk,
        )

        def read_block(block_idx: int) -> np.ndarray:
            _, rows, _, cols = tiles[block_idx]
            if use_counts_t:
                return _read_facade_block(zarr_arr, cols, rows).T
            return _read_facade_block(zarr_arr, rows, cols)

        def accumulate_block(
            raw: np.ndarray,
            local_cells: np.ndarray,
            local_feats: np.ndarray,
        ) -> None:
            height = raw.shape[0]
            width = raw.shape[1]
            for row_start in range(0, height, sub_rows):
                row_end = min(height, row_start + sub_rows)
                local_inv = inv_scalar[local_cells[row_start:row_end]]
                for col_start in range(0, width, sub_cols):
                    col_end = min(width, col_start + sub_cols)
                    band = raw[row_start:row_end, col_start:col_end]
                    feat_slice = local_feats[col_start:col_end]
                    nz[feat_slice] += (band > 0).sum(axis=0)
                    scaled = band.astype(np.float64, copy=True)
                    scaled *= sf
                    np.multiply(scaled, local_inv[:, None], out=scaled)
                    s1[feat_slice] += scaled.sum(axis=0)
                    s2[feat_slice] += np.einsum("ij,ij->j", scaled, scaled)

        for block_idx, raw, read_sec, source in iter_column_blocks(
            n_blocks,
            read_block,
            remote=remote,
            msg=f"({self.name}) Computing feature stats",
        ):
            local_cells, _, local_feats, _ = tiles[block_idx]
            t_compute = time.perf_counter()
            accumulate_block(raw, local_cells, local_feats)
            compute_sec = time.perf_counter() - t_compute
            logger.info(
                f"({self.name}) feature stats block {block_idx + 1}/{n_blocks}: "
                f"read {read_sec:.1f}s ({source}) compute {compute_sec:.1f}s "
                f"rss {process_rss_mb():.0f} MiB"
            )
            del raw

        mean = s1 / n_cells
        sigmas = s2 / n_cells - np.square(mean)
        return {"normed_tot": s1, "normed_n": nz, "sigmas": sigmas}

    def set_feature_stats(self, cell_key: str) -> None:
        """Calculates summary statistics for the features of the assay using
        only cells that are marked True by the 'cell_key' parameter.

        Args:
            cell_key: Name of the key (column) from cell attribute table.

        Returns: None
        """
        feat_key = "I"  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is True:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        else:
            if identifier in self.feats.locations:
                del self.feats.locations[identifier]
        n_used = int(len(cell_idx))
        # The single-pass streaming path only implements the library-size
        # normalization formula (sf * raw / nCounts). Any other norm method
        # (e.g. log-transformed or renormalized variants) falls back to the
        # generic ChunkedArray reductions, which honour self.normMethod.
        if self.normMethod is norm_lib_size:
            stats = self._streaming_feature_stats(cell_idx, feat_idx)
            n_cells = stats["normed_n"]
            tot = stats["normed_tot"]
            sigmas = stats["sigmas"]
        else:
            n_cells = show_dask_progress(
                (self.normed(cell_idx, feat_idx) > 0).sum(axis=0),
                f"({self.name}) Computing nCells",
                self.nthreads,
            )
            tot = show_dask_progress(
                self.normed(cell_idx, feat_idx).sum(axis=0),
                f"({self.name}) Computing normed_tot",
                self.nthreads,
            )
            sigmas = show_dask_progress(
                self.normed(cell_idx, feat_idx).var(axis=0),
                f"({self.name}) Computing sigmas",
                self.nthreads,
            )
        # idx = n_cells > min_cells
        # self.feats.update_key(idx, key=feat_key)
        # n_cells, tot, sigmas = n_cells[idx], tot[idx], sigmas[idx]

        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(
            as_zarr_group(self.z[stats_loc], name=stats_loc), identifier
        )
        self.feats.insert(
            "normed_tot", tot.astype(float), overwrite=True, location=identifier
        )
        # Mean over the cells actually used (cell_key subset), matching the
        # denominator of the variance computed above. self.cells.N counts all
        # primary cells, including those filtered out, so it is not used here.
        self.feats.insert(
            "avg",
            (tot / max(1, n_used)).astype(float),
            overwrite=True,
            location=identifier,
        )
        nz_mean = np.divide(
            tot, n_cells, out=np.zeros_like(tot).astype(float), where=n_cells != 0
        )
        self.feats.insert(
            "nz_mean",
            nz_mean.astype(float),
            overwrite=True,
            location=identifier,
        )
        self.feats.insert(
            "sigmas", sigmas.astype(float), overwrite=True, location=identifier
        )
        self.feats.insert(
            "normed_n", n_cells.astype(float), overwrite=True, location=identifier
        )
        self.z[stats_loc].attrs["subset_hash"] = self._create_subset_hash(
            cell_idx, self.feats.active_index(feat_key)
        )
        self.feats.unmount_location(identifier)
        return None

    def get_feature_stats(
        self,
        cell_key: str,
        columns: Sequence[str] | None = None,
        *,
        feat_key: str = "I",
    ) -> dict[str, np.ndarray]:
        """Return cached feature statistics aligned to ``feat_key``.

        This method only reads an existing, valid summary-statistics group. It
        does not calculate statistics or delete a stale cache. Default reads
        prefer adaptive corrected variance and fall back to legacy fixed caches.
        """
        requested: tuple[str, ...] | None
        if columns is None:
            requested = None
        elif isinstance(columns, str):
            raise TypeError("columns must be a sequence of column names, not a string")
        else:
            requested = tuple(columns)
        if requested is not None and not all(
            isinstance(column, str) for column in requested
        ):
            raise TypeError("columns must contain only strings")

        cell_idx, all_feat_idx = self._get_cell_feat_idx(cell_key, "I")
        _, stats_loc = self._get_summary_stats_loc(cell_key)
        if not self._validate_stats_loc(
            stats_loc,
            cell_idx,
            all_feat_idx,
            delete_on_fail=False,
        ):
            raise KeyError(
                f"Summary statistics have not been calculated for cell key: {cell_key}"
            )

        stats_group = as_zarr_group(self.z[stats_loc], name=stats_loc)
        if requested is None:
            adaptive = _corrected_variance_column(200, 0.1, "adaptive")
            fixed = _corrected_variance_column(200, 0.1, "fixed")
            c_var_col = (
                adaptive
                if adaptive in stats_group or fixed not in stats_group
                else fixed
            )
            requested = ("nz_mean", c_var_col, "normed_n")
        feat_idx = self.feats.active_index(feat_key)
        values: dict[str, np.ndarray] = {}
        for column in requested:
            if column not in stats_group:
                raise KeyError(
                    f"Feature statistic {column!r} is not available for cell key "
                    f"{cell_key!r}"
                )
            array = as_zarr_array(stats_group[column], name=f"{stats_loc}/{column}")
            values[column] = np.asarray(array[:])[feat_idx]
        return values

    def set_summary_stats(
        self,
        cell_key: str | None = None,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        *,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
    ) -> tuple[str, str]:
        """Calculates summary statistics for the features of the assay using only cells that are marked True by the 'cell_key' parameter.

        Args:
            cell_key: Name of the key (column) from cell attribute table.
            n_bins: Number of bins to divide the data into.
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
            bin_strategy: Strategy used to construct bins and variance anchors.

        Returns:
            A tuple of two strings.
            identifier: The text that will be prepended to column names when summary statistics are loaded onto the feature attributes table.
            c_var_col: The name of the column in the feature attribute table that contains the corrected variance values.
        """

        def col_renamer(x: str) -> str:
            return f"{identifier}_{x}"

        if cell_key is None:
            cell_key = "I"

        c_var_col = _corrected_variance_column(
            n_bins,
            lowess_frac,
            bin_strategy,
        )
        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        if col_renamer(c_var_col) in self.feats.columns:
            logger.info("Using existing corrected dispersion values")
        else:
            slots = ["normed_tot", "avg", "nz_mean", "sigmas", "normed_n"]
            for i in slots:
                i = col_renamer(i)
                if i not in self.feats.columns:
                    raise KeyError(f"ERROR: {i} not found in feature metadata")
            from ..features.variability import fit_lowess

            mean = self.feats.fetch(col_renamer("avg")).astype(float)
            variance = self.feats.fetch(col_renamer("sigmas")).astype(float)
            positive = mean > 0
            c_var = np.zeros(mean.shape, dtype=float)
            c_var[positive] = fit_lowess(
                mean[positive],
                variance[positive],
                n_bins,
                lowess_frac,
                bin_strategy=bin_strategy,
            )
            self.feats.insert(c_var_col, c_var, overwrite=True, location=identifier)

        return identifier, c_var_col

    def set_hvgs(
        self,
        cell_key: str,
        *,
        mask: np.ndarray | None = None,
        feature_indexes: Sequence[int] | None = None,
        hvg_key_name: str = "hvgs",
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        blacklist: str | None = None,
        blacklist_exclusions: str | None = None,
        blacklist_indexes: Sequence[int] | None = None,
    ) -> str:
        """Install a supplied HVG selection and ensure its summary statistics."""
        if (mask is None) == (feature_indexes is None):
            raise ValueError("Provide exactly one of mask or feature_indexes")

        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("mask must be a NumPy array")
            if mask.shape != (self.feats.N,):
                raise ValueError(f"mask must have shape ({self.feats.N},)")
            if mask.dtype != bool:
                raise TypeError("mask must have boolean dtype")
            selected = mask.copy()
        else:
            assert feature_indexes is not None
            indexes = _as_feature_indexes(
                feature_indexes,
                n_features=self.feats.N,
                name="feature_indexes",
                require_unique=True,
            )
            selected = self.feats.index_to_bool(indexes)

        if not selected.any():
            raise ValueError("HVG selection must contain at least one feature")

        blocked = np.empty(0, dtype=np.int64)
        if blacklist_indexes is not None:
            blocked = _as_feature_indexes(
                blacklist_indexes,
                n_features=self.feats.N,
                name="blacklist_indexes",
                require_unique=False,
            )
        elif blacklist is not None:
            blocked_names = (
                set() if blacklist == "" else set(self.feats.grep(blacklist))
            )
            if blacklist_exclusions is None or blacklist_exclusions == "":
                excluded_names: set[str] = set()
            else:
                excluded_names = set(self.feats.grep(blacklist_exclusions))
            blocked = self.feats.get_index_by(
                sorted(blocked_names - excluded_names),
                "names",
            ).astype(np.int64, copy=False)
        elif blacklist_exclusions not in (None, ""):
            raise ValueError("blacklist_exclusions requires blacklist")

        self.set_summary_stats(
            cell_key,
            n_bins,
            lowess_frac,
            bin_strategy=bin_strategy,
        )
        selected[blocked] = False
        column_name = f"{cell_key}__{hvg_key_name}"
        self.feats.insert(column_name, selected, fill_value=False, overwrite=True)
        return column_name

    # maybe we should return plot here? If one wants to modify it. /raz
    def mark_hvgs(
        self,
        cell_key: str,
        min_cells: int,
        top_n: int,
        min_var: float,
        max_var: float,
        min_mean: float,
        max_mean: float,
        n_bins: int,
        lowess_frac: float,
        blacklist: str,
        hvg_key_name: str,
        keep_bounds: bool,
        show_plot: bool,
        max_cells: int | float,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        **plot_kwargs: Any,
    ) -> None:
        """Identifies highly variable genes in the dataset.

        The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly
        variable. The variance is corrected by first dividing genes into bins based on their mean expression values.
        The fixed strategy fits a Lowess curve through minimum-variance genes, while the adaptive strategy uses balanced
        bins and robust variance anchors. mark_hvgs will by default run on the default assay.
        See `scarf.features.fit_lowess` for further details.

        *Modifies the feats table*: adds a column named `<cell_key>__hvgs` to the feature table,
        which contains a True value for genes marked HVGs. The prefix comes from the `cell_key` parameter,
        the naming rule in Scarf dictates that cells used to identify HVGs are prepended to the column name
        (with a double underscore delimiter).

        Args:
            cell_key: Specify which cells to use to identify the HVGs. (Default value 'I' use all non-filtered out
                      cells).
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to higher signal to noise
                       ratio in the selected features.
            max_cells: Maximum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. This can be useful to filter out genes that are
                       expressed in too many cells. Default value is infinity, meaning no upper limit.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter.
            min_var: Minimum variance threshold for HVG selection.
            max_var: Maximum variance threshold for HVG selection.
            min_mean: Minimum mean value of expression threshold for HVG selection.
            max_mean: Maximum mean value of expression threshold for HVG selection.
            n_bins: Number of bins into which the mean expression is binned.
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
            bin_strategy: Strategy used to construct bins and variance anchors.
            blacklist: A regular expression string pattern. Gene names matching to this pattern will be excluded from
                       the final highly variable genes list
            hvg_key_name: The label for highly variable genes. This label will be used to mark the HVGs in the
                          feature attribute table. The value for 'cell_key' parameter is prepended to this value.
            keep_bounds: If True, then the boundary values are retained and not filtered out.
            show_plot: If True, a plot is produced, that for each gene shows the corrected variance on the y-axis and
                       the non-zero mean (means from cells where the gene had a non-zero value) on the x-axis. The
                       genes are colored in two gradients which indicate the number of cells where the gene was
                       expressed. The colors are yellow to dark red for HVGs, and blue to green for non-HVGs.
            **plot_kwargs: Keyword arguments for ``scarf.plotting.highly_variable_features``
                           (for example ``figsize``, ``label_size``, ``point_sizes``,
                           ``colormaps``).
        """

        def col_renamer(x: str) -> str:
            return f"{identifier}_{x}"

        logger.info("Calculating summary statistics")
        identifier, c_var_col = self.set_summary_stats(
            cell_key,
            n_bins,
            lowess_frac,
            bin_strategy=bin_strategy,
        )
        logger.info("Calculating HVGs")

        from ..features.variability import select_highly_variable_features

        hvgs = select_highly_variable_features(
            corrected_variance=self.feats.fetch_all(col_renamer(c_var_col)),
            normalized_cell_counts=self.feats.fetch_all(col_renamer("normed_n")),
            mean_nonzero=self.feats.fetch_all(col_renamer("nz_mean")),
            active_features=self.feats.fetch_all("I"),
            feature_names=self.feats.fetch_all("names"),
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
        hvg_key_name = cell_key + "__" + hvg_key_name
        logger.info(f"{sum(hvgs)} genes marked as HVGs")
        self.feats.insert(hvg_key_name, hvgs, fill_value=False, overwrite=True)

        if show_plot:
            from ..plotting import highly_variable_features

            nzm, vf, nc = [
                self.feats.fetch(x)
                for x in [col_renamer("nz_mean"), col_renamer(c_var_col), "nCells"]
            ]
            highly_variable_features(
                mean_nonzero=nzm,
                corrected_variance=vf,
                n_cells=nc,
                selected=self.feats.fetch(hvg_key_name),
                show=True,
                **plot_kwargs,
            )

        return None
