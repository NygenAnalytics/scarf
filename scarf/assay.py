"""
- Classes:
    - Assay: A generic Assay class that contains methods to calculate feature level statistics.
    - RNAassay: This assay is designed for feature selection and normalization of scRNA-Seq data.
    - ATACassay: This assay is designed for ATAC-Seq data. It uses TF-IDF normalization and
                 performs feature selection by marking most prevalent peaks.
    - ADTassay: This assay is designed for ADT data (surface antibodies) obtained from CITE-Seq
                experiments. It performs CLR normalization of the data but does not have any
                method for feature selection.
"""

from collections.abc import Callable, Generator
from typing import Any, cast

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, vstack

from ._types import as_zarr_array, as_zarr_group
from .chunked import ChunkedArray
from .metadata import MetaData
from .utils import controlled_compute, logger, show_dask_progress

__all__ = ["Assay", "RNAassay", "ATACassay", "ADTassay"]

type NormMethod = Callable[["Assay", ChunkedArray], ChunkedArray]
type PercentFeatures = dict[str, str]


def _read_block(
    zarr_arr: zarr.Array,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Read ``zarr_arr[row_idx, col_idx]`` returning rows/cols in index order.

    A basic slice is used only for an index run that is provably contiguous
    (consecutive ascending integers), so it selects exactly the requested
    positions and never includes neighbouring rows or columns. Any other
    selection falls back to orthogonal (fancy) indexing, which preserves the
    order of the index arrays. This centralizes the read path so callers never
    hand-roll ``slice(idx[0], idx[-1] + 1)``.
    """
    from .chunked import _is_contiguous

    def axis_sel(idx: np.ndarray) -> slice | np.ndarray:
        idx = np.asarray(idx)
        if idx.size > 0 and _is_contiguous(idx):
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return idx

    row_sel = axis_sel(row_idx)
    col_sel = axis_sel(col_idx)
    if isinstance(row_sel, slice) and isinstance(col_sel, slice):
        return np.asarray(zarr_arr[row_sel, col_sel])
    return np.asarray(zarr_arr.get_orthogonal_selection((row_sel, col_sel)))


def norm_dummy(_: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """A dummy normalizer. Doesn't perform any normalization. This is useful
    when the 'raw data' is already normalized.

    Args:
        _:
        counts: A chunked array with 'raw' counts data

    Returns: A chunked array
    """
    return counts


def norm_lib_size(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs library size normalization on the data. This is the default
    method for RNA assays.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns:  A chunked array (delayed matrix) containing normalized data.
    """
    assert assay.sf is not None and assay.scalar is not None
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def norm_lib_size_log(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs library size normalization and then transforms the values into
    log scale.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    assert assay.sf is not None and assay.scalar is not None
    return cast(ChunkedArray, np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1)))


def norm_clr(_: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs centered log-ratio normalization (ADT). This is the default
    method for ADT assays.

    Args:
        _:
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    f = np.exp(cast(NDArray[Any], np.log1p(counts).sum(axis=0)) / len(counts))
    return cast(ChunkedArray, np.log1p(counts / f))


def norm_tf_idf(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs TF-IDF normalization This is the default method for ATAC
    assays.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    assert (
        assay.n_term_per_doc is not None
        and assay.n_docs is not None
        and assay.n_docs_per_term is not None
    )
    t_f = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(1 + (assay.n_docs / (assay.n_docs_per_term + 1)))
    return t_f * idf


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
    ) -> None:
        self.name = name
        self.cells = cell_data
        self.nthreads = nthreads
        if workspace is None:
            self.rawData = ChunkedArray(
                as_zarr_array(z[f"{name}/counts"], name=f"{name}/counts"),
                nthreads=nthreads,
            )
            self.feats = MetaData(z[f"{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[self.name], name=self.name)
        else:
            self.rawData = ChunkedArray(
                as_zarr_array(
                    z[f"matrices/{name}/counts"], name=f"matrices/{name}/counts"
                ),
                nthreads=nthreads,
            )
            self.feats = MetaData(z[f"{workspace}/{name}/featureData"])  # type: ignore
            self.z = as_zarr_group(z[f"{workspace}/{name}"], name=f"{workspace}/{name}")
        self.attrs = self.z.attrs
        if "percentFeatures" not in self.attrs:
            self.attrs["percentFeatures"] = {}
        self.normMethod: NormMethod = norm_dummy
        self.sf: int | None = None
        self.scalar: np.ndarray | None = None
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None
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
        from .utils import tqdmbar

        sm = None
        for i in tqdmbar(
            self.rawData[self.cells.active_index(cell_key), :].blocks,
            total=self.rawData.numblocks[0],
            desc=f"INFO: Converting raw data from {self.name} assay into CSR format",
        ):
            s = csr_matrix(controlled_compute(i, self.nthreads))
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
            pass
        else:
            ncells = show_dask_progress(
                (self.rawData > 0).sum(axis=0),
                f"({self.name}) Computing nCells and dropOuts",
                self.nthreads,
            )
            self.feats.insert("nCells", ncells, overwrite=True)
            self.feats.insert(
                "dropOuts",
                abs(self.cells.N - self.feats.fetch("nCells")),
                overwrite=True,
            )
            self.feats.update_key(ncells > min_cells, "I")

    def add_percent_feature(self, feat_pattern: str, name: str) -> None:
        """

        Args:
            feat_pattern: A regular expression pattern to identify the features of interest
            name: This will be used as the name of column under which the percentages will
                  be saved

        Returns:

        """
        if name in self._percent_features():
            if self._percent_features()[name] == feat_pattern:
                return None
            else:
                logger.info(f"Pattern for percentage feature {name} updated.")
        percent_features = self._percent_features()
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
        total = show_dask_progress(
            self.rawData[:, feat_idx].sum(axis=1),
            f"({self.name}) Computing {name}",
            self.nthreads,
        )
        if total.sum() == 0:
            logger.warning(
                f"Percentage feature {name} not added because not detected in any cell"
            )
            return None
        self.cells.insert(
            name,
            100 * total / self.cells.fetch_all(self.name + "_nCounts"),
            overwrite=True,
        )

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
    def _create_subset_hash(cell_idx: np.ndarray, feat_idx: np.ndarray) -> int:
        """Takes two index list and hashes them individually and then computes
        hash of the resulting tuple of two hashes. The objective of this
        function is to generate a unique state identifier for the cell and
        feature indices.

        Args:
            cell_idx: Cell row indices
            feat_idx: Feature row indices

        Returns: Returns the final hash
        """
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

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
        batch_size: int,
        location: str,
        log_transform: bool,
        renormalize_subset: bool,
        update_keys: bool,
        mirror: zarr.Array | None = None,
    ) -> ChunkedArray:
        """Create a new zarr group and saves the normalized data in the group
        for the selected features only.

        Args:
            cell_key: Name of the key (column) from cell attribute table. The data will be saved
                      for only those cells that have a True value in this column.
            feat_key: Name of the key (column) from feature attribute table. The data will be saved
                      for only those features that have a True value in this column
            batch_size: Number of cells to store in a single chunk. Higher values lead to larger
                        memory consumption
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

        from .writers import dask_to_zarr

        # FIXME: Extensive documentation needed to justify the naming strategy of slots here
        # Because HVGs and other feature selections have cell key appended in their metadata
        if feat_key != "I":
            feat_key = cell_key + "__" + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        }
        if location in self.z:
            if (
                subset_hash == self.z[location].attrs["subset_hash"]
                and subset_params == self.z[location].attrs["subset_params"]
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
            vals.chunksize,
            self.nthreads,
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

    def iter_normed_feature_wise(
        self,
        cell_key: str | None,
        feat_key: str | None,
        batch_size: int,
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
            batch_size: Number of genes to be loaded in the memory at a time.
            msg: Message to be displayed in the progress bar
            as_dataframe: If true (default) then the yielded matrices are pandas dataframe
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            Generator yielding DataFrames or (matrix, feature index) tuples.
        """
        from .utils import tqdmbar

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
        logger.debug("Will iterate over data of shape: ", data.shape)
        chunks = np.array_split(
            np.arange(0, data.shape[1]), max(1, int(data.shape[1] / batch_size))
        )
        for chunk in tqdmbar(chunks, desc=msg, total=len(chunks)):
            if as_dataframe:
                yield pd.DataFrame(
                    controlled_compute(data[:, chunk], self.nthreads),
                    columns=feat_idx[chunk],
                )
            else:
                yield (
                    controlled_compute(data[:, chunk], self.nthreads).T,
                    feat_idx[chunk],
                )

    def save_normed_for_query(
        self, feat_key: str | None, batch_size: int, overwrite: bool = True
    ) -> None:
        """This methods dumps normalized values for features (as marked by
        `feat_key`) onto disk  in the 'prenormed' slot under the assay's own
        slot.

        Args:
            feat_key: Name of the key (column) from feature attribute table. The data will be fetched
                      for only those features that have a True value in this column. If None then all the features are
                      used
            batch_size: Number of genes to be loaded in the memory at a time.
            overwrite: If True (default value), then will overwrite the existing 'prenormed' slot in the
                       assay hierarchy

        Returns:
            None
        """
        from concurrent.futures import ThreadPoolExecutor

        from .writers import create_zarr_obj_array

        def write_wrapper(idx: str, v: np.ndarray) -> None:
            create_zarr_obj_array(g, idx, v, np.float64, True, False)
            return None

        if "prenormed" in self.z and overwrite is False:
            return None

        g = self.z.create_group("prenormed", overwrite=True)
        for mat, inds in self.iter_normed_feature_wise(
            None, feat_key, batch_size, "Saving features", False
        ):
            write_args = ((inds[i], mat[i]) for i in range(len(inds)))  # type: ignore
            if self.nthreads > 1:
                with ThreadPoolExecutor(max_workers=self.nthreads) as ex:
                    list(ex.map(lambda args: write_wrapper(*args), write_args))
            else:
                for args in write_args:
                    write_wrapper(*args)

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
        batch_size: int = 100,
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
            batch_size: Feature batch size for iteration.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            None
        """

        from .utils import rolling_window
        from .writers import create_zarr_dataset

        cell_ordering = self.cells.fetch(ordering_key, key=cell_key)
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        hashes = [hash(tuple(x)) for x in (cell_idx, feat_idx, cell_ordering)]
        params = {
            "min_exp": min_exp,
            "window_size": window_size,
            "chunk_size": chunk_size,
            "smoothen": smoothen,
            "z_scale": z_scale,
            "norm_params": norm_params,
        }
        location = f"aggregated_{cell_key}_{feat_key}_{ordering_key}"
        if (
            location in self.z
            and "hashes" in self.z[location].attrs
            and hashes == self.z[location].attrs["hashes"]
            and "params" in self.z[location].attrs
            and params == self.z[location].attrs["params"]
        ):
            logger.info(f"Using existing aggregated data from {location}")
        else:
            if location in self.z:
                del self.z[location]

            # The actual size might be smaller due to dynamic filtering of features
            g = create_zarr_dataset(
                self.z,
                location + "/data",
                (batch_size,),
                "float64",
                (feat_idx.shape[0], chunk_size),
            )
            ordering_idx = np.argsort(cell_ordering)
            stored_feat_idx: list[int] = []
            valid_feat_flags: list[bool] = []
            s = 0
            for item in self.iter_normed_feature_wise(
                cell_key,
                feat_key,
                batch_size,
                "Binning over cell-ordering",
                True,
                **norm_params,
            ):
                df = cast(pd.DataFrame, item)
                valid_feat_flags.extend(list((df.mean() > min_exp).values))
                stored_feat_idx.extend(list(df.columns))
                if smoothen:
                    df = rolling_window(df.reindex(ordering_idx).values, window_size)
                if z_scale:
                    df = (df - df.mean(axis=0)) / df.std(axis=0)
                df_mean = np.array(
                    [x.mean(axis=0) for x in np.array_split(df, chunk_size)]
                ).T
                g[s : s + df_mean.shape[0]] = df_mean
                s += df_mean.shape[0]

            g = create_zarr_dataset(
                self.z,
                location + "/feature_indices",
                (len(stored_feat_idx),),
                "uint64",
                (len(stored_feat_idx),),
            )
            g[:] = np.array(stored_feat_idx).astype(int)

            g = create_zarr_dataset(
                self.z,
                location + "/valid_features",
                (len(stored_feat_idx),),
                "bool",
                (len(stored_feat_idx),),
            )
            g[:] = np.array(valid_feat_flags).astype(int)

            self.z[location].attrs["hashes"] = hashes
            self.z[location].attrs["params"] = cast(Any, params)

        ret_val1 = ChunkedArray(
            as_zarr_array(self.z[location + "/data"], name=location + "/data"),
            nthreads=self.nthreads,
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

        from .feat_utils import binned_sampling

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

        from .writers import write_renorm_subset_to_zarr

        if feat_key != "I":
            feat_key = cell_key + "__" + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        }
        if location in self.z:
            if (
                subset_hash == self.z[location].attrs["subset_hash"]
                and subset_params == self.z[location].attrs["subset_params"]
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
        from .storage.budget import worker_prefetch_depth
        from .utils import prefetch_blocks, tqdmbar

        zarr_arr = cast(zarr.Array, self.rawData._backing)
        cell_idx = np.asarray(cell_idx)
        feat_idx = np.asarray(feat_idx)
        scalar_col = np.asarray(scalar, dtype=np.float32).reshape(-1, 1)
        scalar_col[scalar_col == 0] = 1

        batch_size = max(1, batch_size)
        batches = [
            feat_idx[s : s + batch_size] for s in range(0, len(feat_idx), batch_size)
        ]

        def read(cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return _read_block(zarr_arr, cell_idx, cols), cols

        max_ahead = worker_prefetch_depth(requested=max(1, prefetch_depth))

        consumer = prefetch_blocks(batches, read, max_ahead=max_ahead)
        for raw, cols in tqdmbar(consumer, desc=msg or "", total=len(batches)):
            normed = (sf * raw.astype(np.float32)) / scalar_col
            if log_transform:
                normed = np.log1p(normed)
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
        from .storage.budget import worker_prefetch_depth
        from .utils import prefetch_blocks

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
            return start, _read_block(zarr_arr, rows, union)

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

        Reads the raw counts one feature-chunk column block at a time (across
        all selected cells), normalizes each block in float64, and accumulates
        per-feature nonzero count, sum, and sum of squares, so the normalized
        matrix is never fully materialized. The column block follows the array's
        on-disk feature chunk width, and blocks are read ahead in parallel up to
        the worker count. Values match ``norm_lib_size``. Returns ``normed_tot``
        (sum), ``normed_n`` (nonzero count), and ``sigmas`` (population variance).
        """
        from .storage.budget import worker_prefetch_depth
        from .utils import prefetch_blocks, tqdmbar

        zarr_arr = cast(zarr.Array, self.rawData._backing)
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

        chunks = getattr(zarr_arr, "chunks", None)
        col_block = int(chunks[1]) if chunks and len(chunks) > 1 else n_features
        col_block = max(1, min(col_block, n_features))
        col_starts = range(0, n_features, col_block)

        def read(c: int) -> tuple[int, np.ndarray]:
            cols = feat_idx[c : c + col_block]
            return c, _read_block(zarr_arr, cell_idx, cols)

        consumer = prefetch_blocks(col_starts, read, max_ahead=worker_prefetch_depth())
        for c, raw in tqdmbar(
            consumer,
            desc=f"({self.name}) Computing feature stats",
            total=len(col_starts),
        ):
            width = raw.shape[1]
            normed = (sf * raw.astype(np.float64)) * inv_scalar[:, None]
            nz[c : c + width] += (raw > 0).sum(axis=0)
            s1[c : c + width] += normed.sum(axis=0)
            s2[c : c + width] += (normed * normed).sum(axis=0)

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

    def set_summary_stats(
        self, cell_key: str | None = None, n_bins: int = 200, lowess_frac: float = 0.1
    ) -> tuple[str, str]:
        """Calculates summary statistics for the features of the assay using only cells that are marked True by the 'cell_key' parameter.

        Args:
            cell_key: Name of the key (column) from cell attribute table.
            n_bins: Number of bins to divide the data into.
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess

        Returns:
            A tuple of two strings.
            identifier: The text that will be prepended to column names when summary statistics are loaded onto the feature attributes table.
            c_var_col: The name of the column in the feature attribute table that contains the corrected variance values.
        """

        def col_renamer(x: str) -> str:
            return f"{identifier}_{x}"

        if cell_key is None:
            cell_key = "I"

        # check lowess_frac is between 0 and 1
        if not 0 <= lowess_frac <= 1:
            raise ValueError("lowess_frac must be between 0 and 1")

        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        c_var_col = f"c_var__{n_bins}__{lowess_frac}"
        if col_renamer(c_var_col) in self.feats.columns:
            logger.info("Using existing corrected dispersion values")
        else:
            slots = ["normed_tot", "avg", "nz_mean", "sigmas", "normed_n"]
            for i in slots:
                i = col_renamer(i)
                if i not in self.feats.columns:
                    raise KeyError(f"ERROR: {i} not found in feature metadata")
            c_var = self.feats.remove_trend(
                col_renamer("avg"), col_renamer("sigmas"), n_bins, lowess_frac
            )
            self.feats.insert(c_var_col, c_var, overwrite=True, location=identifier)

        return identifier, c_var_col

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
        **plot_kwargs: Any,
    ) -> None:
        """Identifies highly variable genes in the dataset.

        The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly
        variable. The variance is corrected by first dividing genes into bins based on their mean expression values.
        Genes with minimum variance is selected from each bin and a Lowess curve is fitted to
        the mean-variance trend of these genes. mark_hvgs will by default run on the default assay.
        See `utils.fit_lowess` for further details.

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
            blacklist: A regular expression string pattern. Gene names matching to this pattern will be excluded from
                       the final highly variable genes list
            hvg_key_name: The label for highly variable genes. This label will be used to mark the HVGs in the
                          feature attribute table. The value for 'cell_key' parameter is prepended to this value.
            keep_bounds: If True, then the boundary values are retained and not filtered out.
            show_plot: If True, a plot is produced, that for each gene shows the corrected variance on the y-axis and
                       the non-zero mean (means from cells where the gene had a non-zero value) on the x-axis. The
                       genes are colored in two gradients which indicate the number of cells where the gene was
                       expressed. The colors are yellow to dark red for HVGs, and blue to green for non-HVGs.
            **plot_kwargs: Keyword arguments for matplotlib.pyplot.scatter function
        """

        def col_renamer(x: str) -> str:
            return f"{identifier}_{x}"

        logger.info("Calculating summary statistics")
        identifier, c_var_col = self.set_summary_stats(cell_key, n_bins, lowess_frac)
        logger.info("Calculating HVGs")

        if max_mean != np.inf:
            max_mean = 2**max_mean
        if max_var != np.inf:
            max_var = 2**max_var
        if min_mean != -np.inf:
            min_mean = 2**min_mean
        if min_var != -np.inf:
            min_var = 2**min_var

        if blacklist != "":
            bl = self.feats.index_to_bool(
                self.feats.get_index_by(self.feats.grep(blacklist), "names"),
                invert=True,
            )
        else:
            bl = np.ones(self.feats.N).astype(bool)
        if min_var == -np.inf:
            if top_n < 1:
                raise ValueError(
                    "ERROR: Please provide a value greater than 0 for `top_n` parameter"
                )
            idx = self.feats.multi_sift(
                [col_renamer("normed_n"), col_renamer("nz_mean")],
                [min_cells, min_mean],
                [max_cells, max_mean],
                keep_bounds=keep_bounds,
            )
            idx = idx & self.feats.fetch_all("I") & bl
            n_valid_feats = int(idx.sum())
            if n_valid_feats == 0:
                raise ValueError(
                    "No features passed HVG candidate filters "
                    f"(min_cells={min_cells}, max_cells={max_cells}, "
                    f"min_mean={min_mean}, max_mean={max_mean})."
                )
            if top_n >= n_valid_feats:
                logger.warning(
                    f"WARNING: Number of valid features are less then value "
                    f"of parameter `top_n`: {top_n}. Resetting `top_n` to {n_valid_feats - 1}"
                )
                top_n = n_valid_feats - 1
            min_var = (
                pd.Series(self.feats.fetch_all(col_renamer(c_var_col)))[idx]
                .sort_values(ascending=False)
                .values[top_n]
            )  # type: ignore
        hvgs = self.feats.multi_sift(
            [col_renamer(x) for x in ["normed_n", "nz_mean", c_var_col]],
            [min_cells, min_mean, min_var],
            [max_cells, max_mean, max_var],
            keep_bounds=keep_bounds,
        )
        hvgs = hvgs & self.feats.fetch_all("I") & bl
        hvg_key_name = cell_key + "__" + hvg_key_name
        logger.info(f"{sum(hvgs)} genes marked as HVGs")
        self.feats.insert(hvg_key_name, hvgs, fill_value=False, overwrite=True)

        if show_plot:
            from .plots import plot_mean_var

            nzm, vf, nc = [
                self.feats.fetch(x)
                for x in [col_renamer("nz_mean"), col_renamer(c_var_col), "nCells"]
            ]
            plot_mean_var(nzm, vf, nc, self.feats.fetch(hvg_key_name), **plot_kwargs)

        return None


class ATACassay(Assay):
    """This subclass of Assay is designed for feature selection and
    normalization of scATAC-Seq data."""

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
        """This Assay subclass is designed for feature selection and
        normalization of scATAC-Seq data.

        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            **kwargs:

        Attributes:
            normMethod: Pointer to the function to be used for normalization of the raw data
            n_term_per_doc: Number of features per cell. Used for TF-IDF normalization
            n_docs: Number of cells. Used for TF-IDF normalization
            n_docs_per_term: Number of cells per feature. Used for TF-IDF normalization
        """
        super().__init__(
            z=z,
            workspace=workspace,
            name=name,
            cell_data=cell_data,
            nthreads=nthreads,
            min_cells_per_feature=min_cells_per_feature,
            **kwargs,
        )
        self.normMethod = norm_tf_idf
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data. Unlike the `normed` method in the generic Assay
        class this method is optimized for scATAC-Seq data. This method uses
        the normalization indicated by attribute self.normMethod which by
        default is set to `norm_tf_idf`. The TF-IDF normalization is performed
        using only the cells and features indicated by the 'cell_idx' and
        'feat_idx' parameters.

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
        counts: ChunkedArray = self.rawData[:, feat_idx][cell_idx, :]
        self.n_term_per_doc = self.cells.fetch_all(self.name + "_nFeatures")[cell_idx]
        self.n_docs = len(cell_idx)
        self.n_docs_per_term = self.feats.fetch_all("nCells")[feat_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str) -> None:
        """Calculates prevalence of each valid feature of the assay using only
        cells that are marked True by the 'cell_key' parameter. Prevalence of a
        feature is the sum of all its TF-IDF normalized values across cells.

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
        prevalence = show_dask_progress(
            self.normed(cell_idx, feat_idx).sum(axis=0),
            f"({self.name}) Calculating peak prevalence across cells",
            self.nthreads,
        )
        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(
            as_zarr_group(self.z[stats_loc], name=stats_loc), identifier
        )
        self.feats.insert(
            "prevalence", prevalence.astype(float), overwrite=True, location=identifier
        )
        self.z[stats_loc].attrs["subset_hash"] = self._create_subset_hash(
            cell_idx, feat_idx
        )
        self.feats.unmount_location(identifier)
        return None

    def mark_prevalent_peaks(
        self, cell_key: str, top_n: int, prevalence_key_name: str
    ) -> None:
        """Marks `top_n` peaks with highest prevalence as prevalent peaks.

        Args:
           cell_key: Cells to use for selection of most prevalent peaks. The provided value for `cell_key` should be a
                     column in cell attributes table with boolean values.
           top_n: Number of top prevalent peaks to be selected. (Default: 500)
           prevalence_key_name: Base label for marking prevalent peaks in the features attributes column. The value for
                                'cell_key' parameter is prepended to this value.

        Returns: None
        """
        if top_n >= self.feats.N:
            raise ValueError(
                f"ERROR: n_top should be less than total number of features ({self.feats.N})]"
            )
        if isinstance(top_n, int) is False or top_n < 1:
            raise TypeError("ERROR: n_top must a positive integer value")
        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        idx = (
            pd.Series(self.feats.fetch_all(f"{identifier}_prevalence"))
            .sort_values(ascending=False)
            .index.values[:top_n]
        )
        prevalence_key_name = cell_key + "__" + prevalence_key_name
        self.feats.insert(
            prevalence_key_name,
            self.feats.index_to_bool(idx),
            fill_value=False,
            overwrite=True,
        )
        return None


class ADTassay(Assay):
    """This subclass of Assay is designed for normalization of ADT/HTO
    (feature-barcodes library) data from CITE-Seq experiments.

    Args:
        z (zarr.Group): Zarr hierarchy where raw data is located
        name (str): A label/name for assay.
        cell_data: Metadata class object for the cell attributes.
        **kwargs:

    Attributes:
        normMethod: Pointer to the function to be used for normalization of the raw data
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
        """Initialize ADTassay with CLR normalization.

        Args:
            z: Zarr hierarchy where raw data is located.
            name: Assay label.
            cell_data: Cell metadata object.
            **kwargs: Forwarded to ``Assay.__init__`` (workspace, nthreads, etc.).
        """
        super().__init__(
            z=z,
            workspace=workspace,
            name=name,
            cell_data=cell_data,
            nthreads=nthreads,
            min_cells_per_feature=min_cells_per_feature,
            **kwargs,
        )
        self.normMethod = norm_clr

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data. This method uses the normalization indicated
        by attribute self.normMethod which by default is set to `norm_clr`. The
        centered log-ratio normalization is performed using only the cells and
        features indicated by the 'cell_idx' and 'feat_idx' parameters.

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
