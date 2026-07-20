from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..storage.types import as_zarr_group
from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..utils.compute import show_dask_progress
from ..utils.logging import logger
from .base import Assay
from .normalization import norm_tf_idf


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
