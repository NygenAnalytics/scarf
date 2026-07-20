from typing import Any

import numpy as np
import zarr

from ..matrix import ChunkedArray
from ..metadata import MetaData
from .base import Assay
from .normalization import norm_clr


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
