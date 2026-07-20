from typing import Literal

import numpy as np
import zarr

from ..matrix import ChunkedArray

type ZarrArray = zarr.Array
type MatrixData = np.ndarray | ZarrArray | ChunkedArray
type NeighborMetric = Literal["l2", "cosine", "ip"]
