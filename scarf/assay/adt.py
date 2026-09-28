from typing import Any

import zarr

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
            **kwargs,
        )
        self.normMethod = norm_clr
