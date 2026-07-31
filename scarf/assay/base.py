from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, vstack

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import array_digest
from ..utils.compute import controlled_compute, show_dask_progress
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
    statistics. It also provides a method for saving normalized subset of data
    for later KNN graph construction.

    Args:
        z (zarr.Group): Zarr hierarchy where raw data is located
        workspace: Workspace name when assays live under ``matrices/`` (None for legacy layout)
        name (str): A label/name for assay.
        cell_data: Metadata class object for the cell attributes.
        nthreads: number of threads to use for parallel computations
        min_cells_per_feature: Minimum cells expressing a feature for it to be kept

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
        min_cells_per_feature: int = 10,
        matrix_root: zarr.Group | None = None,
        resources: ResourceBudget | None = None,
    ) -> None:
        self.name = name
        self.cells = cell_data
        self.resources = resources or resolve_budget(workers=nthreads)
        self.nthreads = self.resources.workers
        matrix_root = z if matrix_root is None else matrix_root
        if workspace is None:
            counts_path = f"{name}/counts"
            counts_t_path = f"{name}/countsT"
            matrix_group = as_zarr_group(matrix_root[name], name=name)
            self.rawData = ChunkedArray(
                as_zarr_array(matrix_root[counts_path], name=counts_path),
                nthreads=self.nthreads,
                resources=self.resources,
            )
            self.feats = MetaData(z[f"{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[name], name=name)
        else:
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
            self.feats = MetaData(z[f"{workspace}/{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[f"{workspace}/{name}"], name=f"{workspace}/{name}")
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
        if "percentFeatures" not in self.attrs:
            self.attrs["percentFeatures"] = {}
        self.normMethod: NormMethod = norm_dummy
        self.sf: int | None = None
        self.scalar: np.ndarray | None = None
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None
        self._deferred_min_cells_per_feature: int | None = None
        self._ini_feature_props(min_cells_per_feature)

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
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            **kwargs:

        Returns: A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
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

    def _ini_feature_props(self, min_cells: int) -> None:
        """

        Args:
            min_cells: Minimum number of cells per feature. Features below this
                       number are marked invalid.

        Returns:

        """
        if "nCells" in self.feats.columns and "dropOuts" in self.feats.columns:
            return
        if _DEFER_FEATURE_PROPS.get():
            self._deferred_min_cells_per_feature = min_cells
            return
        ncells = show_dask_progress(
            (self.rawData > 0).sum(axis=0),
            f"({self.name}) Computing nCells and dropOuts",
            self.nthreads,
        )
        self._store_feature_props(ncells, min_cells)

    def _store_feature_props(self, ncells: np.ndarray, min_cells: int) -> None:
        self.feats.insert("nCells", ncells, overwrite=True)
        self.feats.insert(
            "dropOuts",
            abs(self.cells.N - self.feats.fetch("nCells")),
            overwrite=True,
        )
        self.feats.update_key(ncells > min_cells, "I")
        self._deferred_min_cells_per_feature = None

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

        row_start = 0
        for raw in self.rawData._stream_blocks(
            nthreads=self.nthreads,
            msg=f"Computing {self.name} initialization statistics",
            prefetch=None,
            row_mask=None,
            resident_bytes=sum(value.nbytes for value in stats.values()),
        ):
            row_stop = row_start + raw.shape[0]
            if compute_n_counts:
                stats["nCounts"][row_start:row_stop] = raw.sum(axis=1)

            positive = None
            if compute_n_features or compute_n_cells:
                positive = raw > 0
            if compute_n_features:
                assert positive is not None
                stats["nFeatures"][row_start:row_stop] = positive.sum(axis=1)
            if compute_n_cells:
                assert positive is not None
                stats["nCells"] += positive.sum(axis=0)

            for name, feat_idx in percent_feature_indices.items():
                stats[name][row_start:row_stop] = raw[:, feat_idx].sum(axis=1)
            row_start = row_stop

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

    def add_percent_feature(self, feat_pattern: str, name: str) -> None:
        """

        Args:
            feat_pattern: A regular expression pattern to identify the features of interest
            name: This will be used as the name of column under which the percentages will
                  be saved

        Returns:

        """
        feat_idx = self._plan_percent_feature(feat_pattern, name)
        if feat_idx is None:
            return None
        total = show_dask_progress(
            self.rawData[:, feat_idx].sum(axis=1),
            f"({self.name}) Computing {name}",
            self.nthreads,
        )
        self._write_percent_feature(name, total)

    def _verify_keys(self, cell_key: str, feat_key: str) -> None:
        """Checks if provided key names are present in cells and feature
        attribute tables and that they are of boolean types.

        Args:
            cell_key: Name of the key (column) from cell attribute table
            feat_key: Name of the key (column) from feature attribute table

        Returns: None

        Note on type checking /GA:
        1. ds.cells.get_dtype(cell_key) == bool returns True because dtype('bool') (from numpy) is conceptually equivalent to Python's bool.
        2. isinstance(ds.cells.get_dtype(cell_key), bool) returns False because dtype('bool') is a numpy.dtype object, not the native Python bool type.
        3. Reason: dtype('bool') is a numpy object, and isinstance checks for the exact class, which is numpy.dtype, not bool.

        """
        if cell_key not in self.cells.columns or self.cells.get_dtype(cell_key) != bool:  # noqa: E721
            raise ValueError(
                f"ERROR: Either {cell_key} does not exist or is not bool type"
            )
        if feat_key not in self.feats.columns or self.feats.get_dtype(feat_key) != bool:  # noqa: E721
            raise ValueError(
                f"ERROR: Either {feat_key} does not exist or is not bool type"
            )

    def _get_cell_feat_idx(
        self, cell_key: str, feat_key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Verifies the provided key by calling _verify_keys and fetches the
        indices of rows that have True value in respective column.

        Args:
            cell_key: Name of the key (column) from cell attribute table
            feat_key: Name of the key (column) from feature attribute table

        Returns: A tuple of two numpy arrays corresponding to cell and feature indices
                 respectively.
        """

        self._verify_keys(cell_key, feat_key)
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return cell_idx, feat_idx

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

    @staticmethod
    def _get_summary_stats_loc(cell_key: str) -> tuple[str, str]:
        """A convenience method that returns the location of feature-wise
        summary statistics Currently summaries are stored under pattern:
        summary_stats_{cell_key}

        Args:
            cell_key: Name of the key (column) from cell attribute table

        Returns: A tuple of two strings. First is the text that will be prepended to column
                 names when summary statistics are loaded onto the feature attributes table. The
                 second is the location of the summary statistics group in the zarr hierarchy of
                 the assay.
        """
        return f"stats_{cell_key}", f"summary_stats_{cell_key}"

    def _validate_stats_loc(
        self,
        stats_loc: str,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        delete_on_fail: bool = True,
    ) -> bool:
        """Check whether the feature-wise summary statistics was previously
        calculated on the same set of features and cells as preset in the
        cell_idx and feat_idx parameters.

        Args:
            stats_loc: Location where the feature summary statistics are saved
            cell_idx: The indices of the cell attribute table
            feat_idx: The indices of the feature attribute table
            delete_on_fail: Whether to delete the summary statistics group if the validity check fails (Default: True).

        Returns: True is the validity test passes otherwise False
        """
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if "subset_hash" in attrs and attrs["subset_hash"] == subset_hash:
                return True
            else:
                # Reset stats loc
                if delete_on_fail:
                    del self.z[stats_loc]
                return False
        else:
            return False

    def _load_stats_loc(self, cell_key: str) -> str:
        """Loads the feature-wise summary statistics calculated on the cells
        that are True in the 'cell_key' column.

        Args:
            cell_key: Name of the key (column) from cell attribute table

        Returns: Location of the group group that contains feature-wise summary statistics
        """
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, "I")
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is False:
            raise KeyError(
                f"Summary statistics have not been calculated for cell key: {cell_key}"
            )
        if identifier not in self.feats.locations:
            self.feats.mount_location(
                as_zarr_group(self.z[stats_loc], name=stats_loc), identifier
            )
        else:
            logger.debug(f"Location ({stats_loc}) already mounted")
        return identifier

    @staticmethod
    def _finalize_staged_mirror(
        mirror: zarr.Array | None,
        subset_hash: str,
        subset_params: dict[str, Any],
    ) -> None:
        """Mark a mirrored staging array complete so staging reuses it as-is."""
        if mirror is None:
            return
        mirror.attrs["staged_subset_hash"] = subset_hash
        mirror.attrs["staged_subset_params"] = subset_params
        mirror.attrs["staged_complete"] = True

    def save_normalized_data(
        self,
        cell_key: str,
        feat_key: str,
        location: str,
        log_transform: bool,
        renormalize_subset: bool,
        update_keys: bool,
        mirror: zarr.Array | None = None,
        artifact_mode: bool = False,
    ) -> ChunkedArray:
        """Create a new zarr group and saves the normalized data in the group
        for the selected features only.

        Args:
            cell_key: Name of the key (column) from cell attribute table. The data will be saved
                      for only those cells that have a True value in this column.
            feat_key: Name of the key (column) from feature attribute table. The data will be saved
                      for only those features that have a True value in this column
            location: Zarr group wherein to save the normalized values
            log_transform: Whether to log transform the values. Is only used if the 'normed' method
                           takes this parameter, ex. RNAassay
            renormalize_subset: Only used if the 'normed' method takes this parameter. Please refer
                                to the documentation of the 'normed' method of the RNAassay for
                                further description of this parameter.
            update_keys: Whether to update the keys. If True then the 'latest_feat_key' and
                         'latest_cell_key' attributes of the assay will be updated. It can be useful
                         to set False in case where you only need to save the normalized data but
                         don't intend to use it directly. For example, when mapping onto a different
                         dataset and aligning features to that dataset.

        Returns: A chunked array containing the normalized data
        """

        from ..storage.materialize import dask_to_zarr

        # FIXME: Extensive documentation needed to justify the naming strategy of slots here
        # Because HVGs and other feature selections have cell key appended in their metadata
        if feat_key != "I":
            feat_key = cell_key + "__" + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params: dict[str, Any] = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        }
        normalization_identity = getattr(self.normMethod, "artifact_identity", None)
        if normalization_identity is not None:
            subset_params["normalization_identity"] = str(normalization_identity)
        if artifact_mode:
            if location not in self.z:
                raise KeyError(f"Artifact group does not exist at {location}")
            if location + "/data" in self.z:
                return ChunkedArray(
                    as_zarr_array(
                        self.z[location + "/data"],
                        name=location + "/data",
                    ),
                    nthreads=self.nthreads,
                    resources=self.resources,
                )
            vals = self.normed(
                cell_idx,
                feat_idx,
                log_transform=log_transform,
                renormalize_subset=renormalize_subset,
            )
            dask_to_zarr(
                vals,
                self.z,
                location + "/data",
                self.nthreads,
                mirror=mirror,
                resources=self.resources,
                stats_group=(
                    as_zarr_group(self.z[location], name=location)
                    if artifact_mode
                    else None
                ),
            )
            return ChunkedArray(
                as_zarr_array(self.z[location + "/data"], name=location + "/data"),
                nthreads=self.nthreads,
                resources=self.resources,
            )
        if location in self.z:
            attrs = self.z[location].attrs
            if (
                attrs.get("subset_hash") == subset_hash
                and attrs.get("subset_params") == subset_params
            ):
                logger.debug(
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
                    resources=self.resources,
                )
            else:
                # Creating group here to overwrite all children
                self.z.create_group(location, overwrite=True)
        vals = self.normed(
            cell_idx,
            feat_idx,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
        )
        dask_to_zarr(
            vals,
            self.z,
            location + "/data",
            self.nthreads,
            mirror=mirror,
            resources=self.resources,
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
            resources=self.resources,
        )

    def iter_normed_feature_wise(
        self,
        cell_key: str | None,
        feat_key: str | None,
        batch_size: int | None,
        msg: str | None,
        as_dataframe: bool = True,
        **norm_params: Any,
    ) -> Generator[pd.DataFrame | tuple[np.ndarray, np.ndarray], None, None]:
        """This generator iterates over all the features marked by `feat_key`
        in batches.

        Args:
            cell_key: Name of the key (column) from cell attribute table. The data will be fetched
                      for only those cells that have a True value in this column. If None then all the cells are used
            feat_key: Name of the key (column) from feature attribute table. The data will be fetched
                      for only those features that have a True value in this column. If None then all the features are
                      used
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
        cell_key: str,
        feat_key: str,
        ordering_key: str,
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
        cell_ordering = np.asarray(
            self.cells.fetch(ordering_key, key=cell_key),
            dtype=float,
        )
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        n_cells = cell_ordering.shape[0]
        if cell_ordering.ndim != 1 or n_cells == 0:
            raise ValueError("Cell ordering must be a non-empty one-dimensional array")
        if not np.isfinite(cell_ordering).all():
            raise ValueError("Cell ordering must contain only finite values")
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
        cell_key: str,
        feat_key: str,
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
            cell_key,
            feat_key,
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
            (max(len(feature_indices), 1),),
            "uint64",
            (len(feature_indices),),
        )
        feature_array[:] = feature_indices
        valid_array = create_zarr_dataset(
            group,
            "valid_features",
            (max(len(valid), 1),),
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

    def save_aggregated_ordering(
        self,
        cell_key: str,
        feat_key: str,
        ordering_key: str,
        min_exp: float = 1e-3,
        window_size: int = 200,
        chunk_size: int = 50,
        smoothen: bool = True,
        z_scale: bool = True,
        batch_size: int | None = None,
        **norm_params: Any,
    ) -> tuple[ChunkedArray, NDArray[Any]]:
        """Bin normalized expression along a cell ordering and cache the result.

        Args:
            cell_key: Boolean column in cell metadata selecting cells.
            feat_key: Boolean column in feature metadata selecting features.
            ordering_key: Cell metadata column with pseudotime or ordering values.
            min_exp: Minimum mean expression to retain a feature.
            window_size: Rolling window size for smoothing along ordering.
            chunk_size: Number of ordering bins stored per feature row.
            smoothen: Whether to apply rolling-window smoothing.
            z_scale: Whether to z-scale values within each feature.
            batch_size: Feature batch size for iteration. When None, selected
                features are grouped into chunk-aligned blocks that fit the
                operation memory budget.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            None
        """
        import warnings

        warnings.warn(
            "Assay.save_aggregated_ordering writes the legacy cache layout; "
            "use DataStore.run_pseudotime_aggregation for artifact-backed persistence.",
            DeprecationWarning,
            stacklevel=2,
        )

        (
            cell_ordering,
            _cell_idx,
            feat_idx,
            effective_window,
            effective_bins,
            hashes,
            params,
        ) = Assay._prepare_aggregated_ordering(
            self,
            cell_key,
            feat_key,
            ordering_key,
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            norm_params=dict(norm_params),
        )
        location = f"aggregated_{cell_key}_{feat_key}_{ordering_key}"

        def _cached_aggregation_valid() -> bool:
            if location not in self.z:
                return False
            group = as_zarr_group(self.z[location], name=location)
            attrs = group.attrs
            if attrs.get("hashes") != hashes or attrs.get("params") != params:
                return False
            if not all(
                name in group for name in ("data", "feature_indices", "valid_features")
            ):
                return False
            data_arr = as_zarr_array(group["data"], name="data")
            feat_arr = as_zarr_array(group["feature_indices"], name="feature_indices")
            valid_arr = as_zarr_array(group["valid_features"], name="valid_features")
            if data_arr.ndim != 2 or data_arr.shape[1] != effective_bins:
                return False
            if (
                feat_arr.shape[0] != data_arr.shape[0]
                or valid_arr.shape[0] != data_arr.shape[0]
            ):
                return False
            return True

        if _cached_aggregation_valid():
            logger.debug(f"Using existing aggregated data from {location}")
        else:
            if location in self.z:
                del self.z[location]
            group = self.z.create_group(location)
            Assay._write_aggregated_ordering_group(
                self,
                group,
                cell_key=cell_key,
                feat_key=feat_key,
                cell_ordering=cell_ordering,
                feat_idx=feat_idx,
                min_exp=min_exp,
                effective_window=effective_window,
                effective_bins=effective_bins,
                smoothen=smoothen,
                z_scale=z_scale,
                batch_size=batch_size,
                norm_params=dict(norm_params),
            )
            group.attrs["hashes"] = hashes
            group.attrs["params"] = cast(Any, params)

        ret_val1 = ChunkedArray(
            as_zarr_array(self.z[location + "/data"], name=location + "/data"),
            nthreads=self.nthreads,
            resources=self.resources,
        )
        ret_val2 = np.asarray(
            as_zarr_array(
                self.z[location + "/feature_indices"],
                name=location + "/feature_indices",
            )[:]
        )

        if location + "/valid_features" in self.z:
            valid_feats = np.asarray(
                as_zarr_array(
                    self.z[location + "/valid_features"],
                    name=location + "/valid_features",
                )[:],
                dtype=bool,
            )
            ret_val1 = ret_val1[valid_feats]
            ret_val2 = ret_val2[valid_feats]

        return ret_val1, ret_val2

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

        cell_idx, _ = self._get_cell_feat_idx(cell_key, "I")
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

        from ..features.scoring import binned_sampling
        from .rna import RNAassay

        def _names_to_idx(i: list[str]) -> np.ndarray:
            return self.feats.get_index_by(i, "names", None)

        def _calc_mean(i: np.ndarray | list[str] | list[int]) -> np.ndarray:
            if isinstance(i, list):
                if not i or isinstance(i[0], str):
                    feat_selection = self.feats.get_index_by(i, "names", None)
                else:
                    feat_selection = np.asarray(i, dtype=int)
            else:
                feat_selection = i
            return (
                self.normed(cell_idx=cell_idx, feat_idx=np.sort(feat_selection))
                .mean(axis=1)
                .compute()
            )

        feature_idx = _names_to_idx(feature_names)
        if len(feature_idx) == 0:
            raise ValueError(
                f"ERROR: No feature ids found for any of the provided {len(feature_names)} features"
            )

        identifier = self._load_stats_loc(cell_key)
        obs_avg = pd.Series(self.feats.fetch_all(f"{identifier}_avg"))
        control_idx = binned_sampling(
            obs_avg, list(feature_idx), ctrl_size, n_bins, rand_seed
        )
        cell_idx, _ = self._get_cell_feat_idx(cell_key, "I")
        if isinstance(self, RNAassay) and self.normMethod is norm_lib_size:
            means = self._mean_normed_feature_groups(
                cell_idx,
                {
                    "target": np.asarray(feature_idx, dtype=int),
                    "control": np.asarray(control_idx, dtype=int),
                },
            )
            return np.asarray(means["target"] - means["control"])
        return np.asarray(_calc_mean(feature_idx) - _calc_mean(control_idx))

    def __repr__(self) -> str:
        f = self.feats.fetch_all("I")
        assay_name = str(self.__class__).split(".")[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"
